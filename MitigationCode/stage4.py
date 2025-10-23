
import os
import json
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
import re

# Configuration paths
MODEL_PATH = "MODEL HERE"  
STAGE1_DIR = "PATH HERE"
STAGE2_DIR = "PATH HERE"
STAGE3_DIR = "PATH HERE"
STAGE4_DIR = "PATH HERE"
DATASET_PATH = "PATH HERE"

os.makedirs(STAGE4_DIR, exist_ok=True)
os.makedirs(os.path.join(STAGE4_DIR, "multiturn_outputs"), exist_ok=True)

def load_a_model():
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        
        print(f"Loading a model: {MODEL_PATH}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Configure quantization for memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            output_hidden_states=True
        )
        
        model.eval()
        print("a model loaded successfully")
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading a model: {e}")
        return None, None

@dataclass
class MaskConfig:
    theta_high: float = 0.7
    theta_low: float = 0.5
    theta_off: float = 0.3
    gamma_high: float = 0.8
    gamma_medium: float = 0.5
    gamma_low: float = 0.2
    alpha: float = 0.5
    k_total: int = 48
    safety_min: float = 0.1
    beta_decay: float = 0.9

class Stage1BiasLoader:
    """Loads Stage-1 bias scores from CSV"""
    
    def __init__(self, stage1_dir: str):
        self.stage1_dir = stage1_dir
        self.bias_scores = self.load_stage1_scores()
    
    def load_stage1_scores(self) -> Dict[str, float]:
        csv_file = os.path.join(self.stage1_dir, "stage1_detailed_results.csv")
        
        if not os.path.exists(csv_file):
            print(f"Warning: Stage-1 CSV not found at {csv_file}")
            return {}
        
        try:
            df = pd.read_csv(csv_file)
            print(f"Loaded Stage-1 results: {len(df)} rows")
            
            bias_dict = {}
            for _, row in df.iterrows():
                conv_id = int(row['conversation_id'])
                turn_num = int(row['turn_number'])
                bias_score = float(row['overall_bias_score'])
                key = f"item_{conv_id}_turn{turn_num}"
                bias_dict[key] = bias_score
            
            print(f"Loaded {len(bias_dict)} bias scores from Stage-1")
            return bias_dict
            
        except Exception as e:
            print(f"Error loading Stage-1 CSV: {e}")
            return {}
    
    def get_bias_score(self, item_id: str, turn_id: int = None) -> float:
        if turn_id is not None:
            key = f"{item_id}_turn{turn_id}"
            if key in self.bias_scores:
                return self.bias_scores[key]
        
        if item_id in self.bias_scores:
            return self.bias_scores[item_id]
        
        if turn_id is not None:
            base_score = 0.3 + (turn_id * 0.1)
            return min(base_score, 0.8)
        
        return 0.5

class Stage3MemoryLoader:
    """Loads Stage-3 memory scores (C_t) from MCP JSON"""
    
    def __init__(self, stage3_dir: str):
        self.stage3_dir = stage3_dir
        self.memory_scores = self.load_stage3_scores()
    
    def load_stage3_scores(self) -> Dict[str, float]:
        memory_file = os.path.join(self.stage3_dir, "mcp_scores.json")
        
        if not os.path.exists(memory_file):
            print(f"Warning: Stage-3 MCP scores not found at {memory_file}")
            return {}
        
        try:
            with open(memory_file, 'r') as f:
                mcp_data = json.load(f)
            
            memory_dict = {}
            for key, value in mcp_data.items():
                parts = key.split('_')
                if len(parts) >= 4:
                    conv_num = parts[1]
                    turn_num = parts[3]
                    new_key = f"item_{conv_num}_turn{turn_num}"
                    C_t = value.get('C_t', 0.5)
                    memory_dict[new_key] = C_t
            
            print(f"Loaded {len(memory_dict)} memory scores from Stage-3")
            return memory_dict
            
        except Exception as e:
            print(f"Error loading Stage-3 MCP JSON: {e}")
            return {}
    
    def get_memory_score(self, item_id: str, turn_id: int = None) -> float:
        if turn_id is not None:
            if turn_id == 0:
                return 0.21
            
            key = f"{item_id}_turn{turn_id}"
            if key in self.memory_scores:
                return self.memory_scores[key]
        
        if item_id in self.memory_scores:
            return self.memory_scores[item_id]
        
        if turn_id is not None:
            if turn_id == 0:
                return 0.21
            memory = 0.2 + (turn_id * 0.15)
            return min(memory, 0.9)
        
        return 0.5

class PipelineGatingPolicy:
    """Gating policy: θt = f(St, Ct)"""
    
    def __init__(self, config: MaskConfig, stage1_loader: Stage1BiasLoader, 
                 stage3_loader: Stage3MemoryLoader):
        self.config = config
        self.stage1_loader = stage1_loader
        self.stage3_loader = stage3_loader
        self.current_gate_level = "OFF"
        self.conversation_history = []
        self.cumulative_bias = 0.0
    
    def reset_conversation(self):
        self.current_gate_level = "OFF"
        self.conversation_history = []
        self.cumulative_bias = 0.0
    
    def compute_gating_signal(self, St: float, Ct: float, turn_id: int) -> Tuple[str, float]:
        if turn_id == 0:
            self.cumulative_bias = St
        else:
            self.cumulative_bias = St + self.config.beta_decay * self.cumulative_bias
        
        theta_t = St * (1.0 + Ct)
        
        if theta_t >= self.config.theta_high:
            gate_level = "HIGH"
            intensity = self.config.gamma_high
        elif theta_t >= self.config.theta_low:
            gate_level = "MEDIUM"
            intensity = self.config.gamma_medium
        elif theta_t >= self.config.theta_off:
            gate_level = "LOW"
            intensity = self.config.gamma_low
        else:
            gate_level = "OFF"
            intensity = 0.0
        
        self.current_gate_level = gate_level
        
        self.conversation_history.append({
            "turn": turn_id,
            "St": St,
            "Ct": Ct,
            "theta_t": theta_t,
            "cumulative_bias": self.cumulative_bias,
            "gate_level": gate_level,
            "intensity": intensity
        })
        
        return gate_level, intensity
    
    def should_mask(self, item_id: str, turn_id: int) -> Tuple[bool, float, str, Dict]:
        St = self.stage1_loader.get_bias_score(item_id, turn_id)
        Ct = self.stage3_loader.get_memory_score(item_id, turn_id)
        gate_level, intensity = self.compute_gating_signal(St, Ct, turn_id)
        
        should_mask = (gate_level != "OFF")
        
        metadata = {
            "St_bias": St,
            "Ct_memory": Ct,
            "cumulative_bias": self.cumulative_bias,
            "gate_level": gate_level
        }
        
        return should_mask, intensity, gate_level, metadata

class Stage4MaskBuilder:
    """Builds mask sets from Stage-2 CBS neurons"""
    
    def __init__(self, stage2_dir: str, config: MaskConfig):
        self.stage2_dir = stage2_dir
        self.config = config
        self.local_cbs = self.load_json("cbs_local.json")
        self.carry_cbs = self.load_json("cbs_carry.json")
        
        print(f"Loaded CBS neurons: {len(self.local_cbs)} local, {len(self.carry_cbs)} carry")
    
    def load_json(self, filename: str) -> List[Dict]:
        filepath = os.path.join(self.stage2_dir, filename)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            return []
    
    def build_mask_sets(self) -> Dict[str, Any]:
        carry_ranked = sorted(self.carry_cbs, 
                            key=lambda x: x.get("abs_attribution", 0), 
                            reverse=True)
        local_ranked = sorted(self.local_cbs,
                            key=lambda x: x.get("abs_attribution", 0),
                            reverse=True)
        
        k_carry = min(self.config.k_total, len(carry_ranked))
        k_local = min(20, len(local_ranked))
        k_core = int(k_carry * self.config.alpha)
        k_support = k_carry - k_core
        
        mask_plan = {
            "config": {
                "k_carry": k_carry,
                "k_core": k_core,
                "k_support": k_support,
                "k_local": k_local
            },
            "core_neurons": carry_ranked[:k_core],
            "support_neurons": carry_ranked[k_core:k_carry],
            "local_neurons": local_ranked[:k_local],
            "all_neurons": carry_ranked[:k_carry] + local_ranked[:k_local]
        }
        
        mask_file = os.path.join(STAGE4_DIR, "mask_plan_pipeline.json")
        with open(mask_file, 'w') as f:
            json.dump(mask_plan, f, indent=2, default=str)
        
        print(f"Mask plan saved with {k_carry + k_local} total neurons")
        return mask_plan

class NeuronMasker:
    """Applies dynamic adaptive suppression"""
    
    def __init__(self, mask_plan: Dict[str, Any], config: MaskConfig):
        self.mask_plan = mask_plan
        self.config = config
        self.hooks = []
        self.current_intensity = 0.0
        self.current_gate_level = "OFF"
        self.prepare_mask_indices()
    
    def prepare_mask_indices(self):
        self.mask_indices = defaultdict(lambda: {"core": [], "support": [], "local": []})
        
        for neuron in self.mask_plan["core_neurons"]:
            self.mask_indices[neuron["layer"]]["core"].append(neuron["neuron"])
        for neuron in self.mask_plan["support_neurons"]:
            self.mask_indices[neuron["layer"]]["support"].append(neuron["neuron"])
        for neuron in self.mask_plan["local_neurons"]:
            self.mask_indices[neuron["layer"]]["local"].append(neuron["neuron"])
    
    def create_masking_hook(self, layer_id: int):
        def masking_hook(module, input, output):
            if self.current_intensity == 0:
                return output
            
            layer_masks = self.mask_indices[layer_id]
            if not any(layer_masks.values()):
                return output
            
            if isinstance(output, tuple):
                h = output[0].clone()
            else:
                h = output.clone()
            
            def safe_suppress(base_gamma):
                return max(1.0 - base_gamma * self.current_intensity, self.config.safety_min)
            
            try:
                if layer_masks["core"]:
                    core_indices = torch.tensor(layer_masks["core"], device=h.device)
                    valid_core = core_indices[core_indices < h.size(-1)]
                    if len(valid_core) > 0:
                        h[..., valid_core] *= safe_suppress(1.0)
                
                if layer_masks["support"]:
                    support_indices = torch.tensor(layer_masks["support"], device=h.device)
                    valid_support = support_indices[support_indices < h.size(-1)]
                    if len(valid_support) > 0:
                        h[..., valid_support] *= safe_suppress(0.7)
                
                if layer_masks["local"]:
                    local_indices = torch.tensor(layer_masks["local"], device=h.device)
                    valid_local = local_indices[local_indices < h.size(-1)]
                    if len(valid_local) > 0:
                        h[..., valid_local] *= safe_suppress(0.5)
            
            except Exception as e:
                print(f"Error in masking hook layer {layer_id}: {e}")
                return output
            
            if isinstance(output, tuple):
                return (h,) + output[1:]
            else:
                return h
        
        return masking_hook
    
    def register_hooks(self, model):
        self.model = model
        
        # a uses model.layers directly
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            transformer_layers = model.model.layers
        elif hasattr(model, 'layers'):
            transformer_layers = model.layers
        else:
            print("Warning: Could not find transformer layers")
            return
        
        for layer_id in self.mask_indices.keys():
            if layer_id < len(transformer_layers):
                hook = self.create_masking_hook(layer_id)
                handle = transformer_layers[layer_id].register_forward_hook(hook)
                self.hooks.append(handle)
        
        print(f"Registered {len(self.hooks)} masking hooks on a layers")
    
    def set_masking_state(self, intensity: float, gate_level: str = "OFF"):
        self.current_intensity = intensity
        self.current_gate_level = gate_level
    
    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

@dataclass
class ConversationTurn:
    turn_id: int
    user_message: str
    assistant_response_baseline: str = ""
    assistant_response_masked: str = ""
    St_bias: float = 0.0
    Ct_memory: float = 0.0
    cumulative_bias: float = 0.0
    gate_level: str = "OFF"
    masking_intensity: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class Conversation:
    conversation_id: str
    item_id: str
    turns: List[ConversationTurn] = field(default_factory=list)
    conversation_history_baseline: List[Dict[str, str]] = field(default_factory=list)
    conversation_history_masked: List[Dict[str, str]] = field(default_factory=list)
    
    def add_turn(self, turn: ConversationTurn):
        self.turns.append(turn)
    
    def get_baseline_context(self) -> str:
        context = []
        for msg in self.conversation_history_baseline:
            context.append(f"User: {msg['user']}")
            if msg.get('assistant'):
                context.append(f"Assistant: {msg['assistant']}")
        return "\n".join(context)
    
    def get_masked_context(self) -> str:
        context = []
        for msg in self.conversation_history_masked:
            context.append(f"User: {msg['user']}")
            if msg.get('assistant'):
                context.append(f"Assistant: {msg['assistant']}")
        return "\n".join(context)

def parse_conversation_from_item(item: Dict, item_idx: int) -> Tuple[str, List[Dict[str, Any]]]:
    turn_pattern = re.compile(r'(\d+)-turn\s+Conv', re.IGNORECASE)
    turns_dict = {}
    
    for key, value in item.items():
        match = turn_pattern.search(key)
        if match:
            turn_num = int(match.group(1))
            if isinstance(value, str):
                turns_dict[turn_num] = value.strip()
    
    conversation = []
    for turn_num in sorted(turns_dict.keys()):
        conversation.append({'turn_id': turn_num, 'prompt': turns_dict[turn_num]})
    
    item_id = f"item_{item_idx}"
    return item_id, conversation

class MultiTurnInferenceEngine:
    """Multi-turn inference with MODEL"""
    
    def __init__(self, model, tokenizer, stage4_dir: str):
        self.model = model
        self.tokenizer = tokenizer
        self.stage4_dir = stage4_dir
    
    def prepare_conversations_from_dataset(self, dataset: List[Dict]) -> List[Tuple[str, List[Dict]]]:
        conversations = []
        for idx, item in enumerate(dataset):
            if isinstance(item, dict):
                item_id, conversation = parse_conversation_from_item(item, idx)
                if conversation:
                    conversations.append((item_id, conversation))
        print(f"Parsed {len(conversations)} conversations")
        return conversations
    
    def format_a_prompt(self, context: str, user_message: str) -> str:
        """Format prompt for model"""
        if context:
            full_context = f"{context}\nUser: {user_message}"
        else:
            full_context = f"User: {user_message}"
        
       
        return f"[INST] {full_context} [/INST]"
    
    def run_multiturn_inference(self, conversations: List[Tuple[str, List[Dict]]], 
                                masker: NeuronMasker,
                                gating_policy: PipelineGatingPolicy,
                                max_conversations: Optional[int] = None) -> List[Conversation]:
        
        if max_conversations:
            conversations = conversations[:max_conversations]
        
        print(f"\nRunning inference on {len(conversations)} conversations with a")
        
        all_conversations = []
        
        for conv_idx, (item_id, conversation_turns) in enumerate(conversations):
            print(f"\nConversation {conv_idx + 1} (ID: {item_id})")
            gating_policy.reset_conversation()
            
            conversation = Conversation(
                conversation_id=f"conv_{conv_idx}",
                item_id=item_id
            )
            
            for turn_dict in conversation_turns:
                turn_id = turn_dict['turn_id']
                prompt = turn_dict['prompt']
                
                should_mask, intensity, gate_level, metadata = gating_policy.should_mask(item_id, turn_id)
                
                print(f"Turn {turn_id}: Gate={gate_level}, Intensity={intensity:.3f}")
                
                baseline_context = conversation.get_baseline_context()
                masked_context = conversation.get_masked_context()
                
                # Baseline generation
                masker.set_masking_state(0.0, "OFF")
                baseline_input = self.format_a_prompt(baseline_context, prompt)
                baseline_response = self.generate_response(baseline_input)
                
                # Masked generation
                masker.set_masking_state(intensity if should_mask else 0.0, gate_level)
                masked_input = self.format_a_prompt(masked_context, prompt)
                masked_response = self.generate_response(masked_input)
                
                turn = ConversationTurn(
                    turn_id=turn_id,
                    user_message=prompt,
                    assistant_response_baseline=baseline_response,
                    assistant_response_masked=masked_response,
                    St_bias=metadata['St_bias'],
                    Ct_memory=metadata['Ct_memory'],
                    cumulative_bias=metadata['cumulative_bias'],
                    gate_level=gate_level,
                    masking_intensity=intensity
                )
                
                conversation.add_turn(turn)
                conversation.conversation_history_baseline.append({"user": prompt, "assistant": baseline_response})
                conversation.conversation_history_masked.append({"user": prompt, "assistant": masked_response})
            
            all_conversations.append(conversation)
        
        return all_conversations
    
    def generate_response(self, formatted_prompt: str, max_new_tokens: int = 100) -> str:
        try:
            inputs = self.tokenizer.encode(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = inputs.to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.5,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    top_p=0.9,
                    repetition_penalty=1.1
                )
            
            response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            return response.strip()
            
        except Exception as e:
            print(f"Generation error: {e}")
            return "[Error]"
    
    def save_results(self, conversations: List[Conversation], config: MaskConfig):
        json_data = {
            "metadata": {
                "model": "MODEL",
                "pipeline_version": "Stage-4 with a",
                "total_conversations": len(conversations),
                "total_turns": sum(len(c.turns) for c in conversations)
            },
            "config": vars(config),
            "conversations": []
        }
        
        for conv in conversations:
            conv_data = {
                "conversation_id": conv.conversation_id,
                "item_id": conv.item_id,
                "turns": [vars(turn) for turn in conv.turns]
            }
            json_data["conversations"].append(conv_data)
        
        json_file = os.path.join(self.stage4_dir, "multiturn_outputs", "a_inference_full.json")
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"Saved results to {json_file}")

class Stage4PipelinePipeline:
    """Complete Stage-4 pipeline"""
    
    def __init__(self, model, tokenizer, config: MaskConfig = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or MaskConfig()
        self.stage1_loader = Stage1BiasLoader(STAGE1_DIR)
        self.stage3_loader = Stage3MemoryLoader(STAGE3_DIR)
        self.mask_builder = Stage4MaskBuilder(STAGE2_DIR, self.config)
        self.inference_engine = MultiTurnInferenceEngine(model, tokenizer, STAGE4_DIR)
    
    def run_complete_pipeline(self, max_conversations: Optional[int] = None) -> bool:
        print("Stage-4 MAM Pipeline with a Model")
        
        try:
            with open(DATASET_PATH, 'r') as f:
                dataset = json.load(f)
            
            conversations = self.inference_engine.prepare_conversations_from_dataset(dataset)
            mask_plan = self.mask_builder.build_mask_sets()
            
            masker = NeuronMasker(mask_plan, self.config)
            masker.register_hooks(self.model)
            
            gating_policy = PipelineGatingPolicy(self.config, self.stage1_loader, self.stage3_loader)
            
            completed_conversations = self.inference_engine.run_multiturn_inference(
                conversations, masker, gating_policy, max_conversations
            )
            
            self.inference_engine.save_results(completed_conversations, self.config)
            
            print(f"\n Pipeline completed with {len(completed_conversations)} conversations")
            return True
            
        except Exception as e:
            print(f"Pipeline error: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    print("Stage-4 MAM with a Model")
    
    model, tokenizer = load_a_model()
    
    if model is None:
        print("Failed to load a model")
        return False
    
    config = MaskConfig(
        theta_high=0.7,
        theta_low=0.5,
        theta_off=0.3,
        gamma_high=0.5,
        gamma_medium=0.3,
        gamma_low=0.15,
        alpha=0.5,
        k_total=48,
        safety_min=0.3,
        beta_decay=0.9
    )
    
    pipeline = Stage4PipelinePipeline(model, tokenizer, config)
    success = pipeline.run_complete_pipeline(max_conversations=None)
    
    if success:
        print("\n Stage-4 completed successfully with a")
    
    return success

if __name__ == "__main__":
    main()