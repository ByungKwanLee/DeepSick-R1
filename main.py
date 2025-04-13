from utils import *
class DatasetParser(Dataset):
    def __init__(self, mode):
        
        # super init
        super().__init__()

        # load the finance data for vision language model - QA pairs
        data = load_dataset("sujet-ai/Sujet-Finance-QA-Vision-100k")

        if mode=='train':
            self.data = data['train']
        else:
            self.data = data['test']
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        if 'image' in self.data[index].keys():
            
            # img_path / instruction
            img_pil = self.data[index]['image']
            conversations = eval(self.data[index]['qa_pairs'])

            # img path -> img
            img_tensor = pil_to_tensor(img_pil)
            
            return {'image': img_tensor, 'conversations': conversations}
        
        else:
            return {'conversations': self.data[index]['conversations']}

class GRPOTrainer(Trainer):
    def __init__(self, save_root, accel):
        # super init
        super().__init__(save_root, accel)

    def compute_loss_and_update(self, inputs, **kwargs):
        """
        Arguments:
        - inputs
        - model
        - ref_model
        - vllm_model
        - vllm_sampling_params
        - processor
        - optimizer
        - scheduler
        - num_gens
        - grpo_iters
        - clip_high_eps
        - clip_low_eps
        - kld_beta
        """
        model = kwargs["model"]
        ref_model = kwargs["ref_model"]
        vllm_model = kwargs["vllm_model"]
        vllm_sampling_params = kwargs["vllm_sampling_params"]
        processor = kwargs["processor"]
        optimizer = kwargs["optimizer"]
        scheduler = kwargs["scheduler"]
        num_gens = kwargs["num_gens"]
        max_new_tokens = kwargs['max_new_tokens']
        temperature = kwargs['temperature']
        grpo_iters = kwargs["grpo_iters"]
        clip_high_eps = kwargs["clip_high_eps"]
        clip_low_eps = kwargs["clip_low_eps"]
        kld_beta = kwargs["kld_beta"]

        # preprocessing for text and image
        prompt_list, image_list, qa_index_list = Utility.preprocess(inputs, processor)


        # vLLM generation and merging
        completion_texts = Utility.vLLM_generation(vllm_model,
                                                   vllm_sampling_params,
                                                   max_new_tokens,
                                                   model,
                                                   self.accel,
                                                   num_gens,
                                                   prompt_list,
                                                   image_list,
                                                   len(prompt_list))
        output_texts = [p + c for p, c in zip(prompt_list * num_gens, completion_texts)]
        
        # postprocessing for text and image
        _inputs = processor(text=prompt_list * num_gens, 
                            images=image_list * num_gens, 
                            padding=True, 
                            return_tensors="pt").to(self.accel.device)
        prompt_length = _inputs.input_ids.shape[1]

        # postprocessing for text and image
        new_prompt_list, new_image_list = Utility.postprocess(inputs * num_gens, processor, qa_index_list * num_gens, completion_texts)
        _new_inputs = processor(text=new_prompt_list, 
                            images=new_image_list, 
                            padding=True,
                            return_tensors="pt").to(self.accel.device)
        
        # prompt + answer
        # just in case for that no answer is given
        if prompt_length==_new_inputs.input_ids.shape[1]: prompt_length-=1
        # [prompt_length mighe be errorneous with +1 or -1 differnece for some samples, but it's fine]
        completion_ids = _new_inputs.input_ids[:, prompt_length:]
        completion_mask = _new_inputs.attention_mask[:, prompt_length:]

        # compute reward
        rewards = Utility.compute_reward(model=vllm_model,
                                         sampling_params=vllm_sampling_params,
                                         temperature=temperature,
                                         processor=processor,
                                         output_texts=output_texts,
                                         answers=[i['conversations'][qa_ind]['answer'] for i, qa_ind in zip(inputs, qa_index_list)] * num_gens,
                                         accel=self.accel)
        rewards = torch.tensor(rewards).float().to(self.accel.device)
        rewards = rewards.view(-1, num_gens, 2)
        avg_reward = rewards.mean(dim=(0,1))
        sum_rewards = rewards.sum(dim=2)
        advantages = ((sum_rewards.view(-1) - sum_rewards.mean(dim=1).repeat_interleave(num_gens)) / (sum_rewards.std(dim=1).repeat_interleave(num_gens) + 1e-4)).unsqueeze(1)

        self.accel.print('----------------Example Generation----------------')
        self.accel.print(output_texts[0])
        self.accel.print('')
        self.accel.print('')
        self.accel.print(f'Reward-Ans: {rewards[0][0][0]}, Reward-Format: {rewards[0][0][1]}')
        self.accel.print(f'Advantage: {advantages[0][0]}')
        self.accel.print(f'Completion mask shape: {completion_mask.shape}')
        self.accel.print(f'Completion shape: {len(completion_texts)}')
        self.accel.print(f'prompt_length: {prompt_length}')
        self.accel.print(f'_new_inputs.input_ids shape: {_new_inputs.input_ids.shape}')
        self.accel.print('--------------------------------------------------')

        # per token logps
        with torch.no_grad():
            old_per_token_logps = Utility.compute_log_probs(model, _new_inputs, completion_ids.shape[1])
            ref_per_token_logps = Utility.compute_log_probs(ref_model, _new_inputs, completion_ids.shape[1])

        # GPU STOP and Memory optimization
        self.memory_optimization()

        # GRPO iterations
        grpo_loss_list = []
        for _ in range(grpo_iters):
            # GRPO Loss per iteration
            new_per_token_logps = Utility.compute_log_probs(model,  _new_inputs, completion_ids.shape[1])
            ratio = torch.exp(new_per_token_logps - old_per_token_logps)
            surrogate_loss = torch.min(ratio * advantages, torch.clamp(ratio, 1-clip_low_eps, 1+clip_high_eps) * advantages)
            kl = torch.exp(ref_per_token_logps - new_per_token_logps) - (ref_per_token_logps - new_per_token_logps) - 1
            per_token_loss = surrogate_loss - kld_beta * kl
            grpo_loss = -((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

            # Backward
            self.accel.backward(grpo_loss)
            optimizer.step()
            optimizer.zero_grad()

            # listing to measure avg of grpo loss
            grpo_loss_list.append(grpo_loss.item())

            # GPU STOP and Memory optimization
            self.memory_optimization()

        # scheduler step
        scheduler.step()

        return {'GRPO-Loss': sum(grpo_loss_list)/len(grpo_loss_list), 'Reward-Ans': avg_reward[0].item(), 'Reward-Format': avg_reward[1].item()}

def train(args):

    # Accelerator for DDP, FSDP, DeepSpeed, etc [Should First Call]
    accel = Accelerator(gradient_accumulation_steps=args.grad_accumul)

    # wandb
    if args.wandb and accel.is_main_process and accel.local_process_index==0:
        wandb.login(key=args.wandb_key)
        wandb.init(project="DeepSick-R1", name=f"DeepSick-R1", dir=os.getcwd(), entity=args.wandb_id)

    # Train Dataset
    train_dataset = DatasetParser('train')
    train_dataloader = DataLoader(train_dataset, 
                                batch_size=args.batch_size, 
                                shuffle=True,
                                num_workers=0,
                                pin_memory=True,
                                collate_fn=lambda x: x)
    
    # Uploading Qwen2.5-3B-VL processor
    min_pixels = 32*28*28
    max_pixels = 512*28*28
    processor = AutoProcessor.from_pretrained(args.model_name, padding_side='left', min_pixels=min_pixels, max_pixels=max_pixels)

    # Uploading Qwen2.5-VL-3B
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    
    # vLLM
    current_device = torch.cuda.current_device()
    vllm_model, vllm_sampling_params = Utility.load_vLLM(args.model_name,
                                                         accel,
                                                         {'temperature':args.temperature,
                                                          'top_p': args.top_p,
                                                          'top_k': args.top_k,
                                                          'max_new_tokens': args.max_new_tokens,
                                                          'repetition_penalty': args.repetition_penalty,
                                                          'max_new_tokens': args.max_new_tokens})
    torch.cuda.set_device(current_device)

    # settings
    for name, param in model.named_parameters():
        if sum([n in name for n in ['self_attn','mlp.up','mlp.down','mlp.gate']]) and 'visual' not in name:
            param.requires_grad=True
        else:
            param.requires_grad=False
                
    # Model Settings
    Utility.bfloat_model(model)
    Utility.bfloat_model(ref_model)
    Utility.freeze_model(ref_model)
    model.train()
    ref_model.eval()

    # setting optimizer and wrapping accelerator
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=args.last_lr/args.lr, total_iters=len(train_dataloader)*args.epochs)
    model, optimizer, scheduler, train_dataloader = accel.prepare(model, optimizer, scheduler, train_dataloader)
    ref_model = accel.prepare(ref_model)

    # GRPOTrainer
    trainer = GRPOTrainer(save_root='./ckpt', accel=accel)
    trainer.train(model=model,
                  ref_model=ref_model,
                  vllm_model=vllm_model,
                  vllm_sampling_params=vllm_sampling_params,
                  epochs=args.epochs,
                  optimizer=optimizer,
                  scheduler=scheduler,
                  processor=processor,
                  train_dataloader=train_dataloader,
                  wandb=args.wandb,
                  num_gens=args.num_gens,
                  temperature=args.temperature,
                  max_new_tokens=args.max_new_tokens,
                  grpo_iters=args.grpo_iters,
                  save_number=args.save_number,
                  clip_high_eps=args.clip_high_eps,
                  clip_low_eps=args.clip_low_eps,
                  kld_beta=args.kld_beta)
    # for name, param in model.named_parameters():
    #     print(f'{name}: {param.dtype} {param.requires_grad}')

if __name__ == "__main__":

    # Argument parameter to be needed
    parser = argparse.ArgumentParser()

    # Wandb
    parser.add_argument('--wandb', default=False, type=Utility.str2bool)
    parser.add_argument('--wandb_key', default="", type=str)
    parser.add_argument('--wandb_id', default="", type=str)

    # model name
    parser.add_argument('--model_name', default="Qwen/Qwen2.5-VL-3B-Instruct", type=str)

    # Training and Saving CKPT Configuration
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--lr', default=5e-6, type=float)
    parser.add_argument('--last_lr', default=1e-6, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--grad_accumul', default=1, type=int)
    parser.add_argument('--save_number', default=10, type=int)
    
    # Generating Answer
    parser.add_argument('--num_gens', default=4, type=int)
    parser.add_argument('--repetition_penalty', default=1.0, type=float)
    parser.add_argument('--temperature', default=0.01, type=float)
    parser.add_argument('--top_p', default=0.001, type=float)
    parser.add_argument('--top_k', default=1, type=int)
    parser.add_argument('--max_new_tokens', default=512, type=int)

    # GRPO Configuration
    parser.add_argument('--grpo_iters', default=4, type=int)
    parser.add_argument('--clip_high_eps', default=0.3, type=float)
    parser.add_argument('--clip_low_eps', default=0.3, type=float)
    parser.add_argument('--kld_beta', default=0.5, type=float)
        
    # argument collection
    args = parser.parse_args()
    
    # Fixing Seed
    Utility.set_all_seeds(42)

    # train
    train(args)