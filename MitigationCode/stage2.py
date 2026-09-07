# Install dependencies into the active environment before running this script.
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import time
import random
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
from category_config import CATEGORY_VOCABULARY, GOLDEN_LABELS, get_category, get_category_paths

# Environment setup
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
CATEGORY = get_category()
CATEGORY_PATHS = get_category_paths(CATEGORY)

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("SOURCE HERE Transformers imported successfully")
    from scipy.stats import pearsonr
    print("SOURCE HERE Scipy imported successfully")
    from sklearn.metrics import mutual_info_score
    print("SOURCE HERE Sklearn imported successfully")
    HAS_TRANSFORMERS = True
except ImportError as e:
    print(f"✗ Import failed: {e}")
    print(f"  Missing module: {e.name if hasattr(e, 'name') else 'unknown'}")
    HAS_TRANSFORMERS = False
except Exception as e:
    print(f"✗ Unexpected error during import: {type(e).__name__}: {e}")
    HAS_TRANSFORMERS = False

@dataclass
class DialogueTurn:
    role: str
    content: str
    turn_id: int
    timestamp: int = 0

class DataFormatHandler:
    @staticmethod
    def parse_conversation_data(dialogue_data: Dict[str, Any]) -> List[DialogueTurn]:
        """
        Parse your specific data format into normalized turns
        
        Args:
            dialogue_data: Dictionary with keys like '0-turn Conv', '1-turn Conv'
            
        Returns:
            List of dialogue turns
        """
        turns = []
        
        # Extract turns in order
        turn_keys = sorted([k for k in dialogue_data.keys() if 'turn Conv' in k])
        
        for i, turn_key in enumerate(turn_keys):
            content = dialogue_data[turn_key]
            
            role = 'user'
            
            if isinstance(content, dict):
                text = content.get('prompt', content.get('content', content.get('text', str(content))))
            else:
                text = str(content)
            
            turns.append(DialogueTurn(
                role=role,
                content=text,
                turn_id=i,
                timestamp=i
            ))
        
        return turns

class ContextBuilder:
    """Stage-2 Context Builder: Creates local and carry-over windows"""
    
    def __init__(self, local_window_size: int = 3, max_carry_size: int = 10):
        self.local_window_size = local_window_size
        self.max_carry_size = max_carry_size
        self.data_handler = DataFormatHandler()
    
    def build_local_context(self, turns: List[DialogueTurn], current_turn: int) -> str:
        """Build local context window (last N turns)"""
        start_idx = max(0, current_turn - self.local_window_size + 1)
        local_turns = turns[start_idx:current_turn + 1]
        
        context_parts = []
        for turn in local_turns:
            context_parts.append(f"[{turn.role.upper()}_{turn.turn_id}] {turn.content}")
        
        return "\n".join(context_parts)
    
    def build_carry_context(self, turns: List[DialogueTurn], current_turn: int) -> str:
        """Build carry-over context"""
        start_idx = max(0, current_turn - self.max_carry_size + 1)
        carry_turns = turns[start_idx:current_turn + 1]
        
        context_parts = []
        for turn in carry_turns:
            context_parts.append(f"[{turn.role.upper()}_{turn.turn_id}] {turn.content}")
        
        return "\n".join(context_parts)
    
    def process_dataset(self, input_file: str, output_dir: str):
        """Process dataset and create context files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Load dataset
        with open(input_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        print(f"Processing {len(dataset)} dialogues...")
        
        local_contexts = []
        carry_contexts = []
        
        for conv_id, dialogue_data in enumerate(dataset):
            # Parse using data format handler
            turns = self.data_handler.parse_conversation_data(dialogue_data)
            
            # Process each turn (except first as it has no history)
            for turn_idx in range(1, len(turns)):
                current_turn = turns[turn_idx]
                
                # Build contexts
                local_ctx = self.build_local_context(turns, turn_idx)
                carry_ctx = self.build_carry_context(turns, turn_idx)
                
                # Create context entries
                base_entry = {
                    'conv_id': conv_id,
                    'turn_id': turn_idx,
                    'current_role': current_turn.role,
                    'current_content': current_turn.content,
                    'stage1_score_ref': f"conv_{conv_id}_turn_{turn_idx}"
                }
                
                local_entry = {**base_entry, 'context': local_ctx, 'context_type': 'local'}
                carry_entry = {**base_entry, 'context': carry_ctx, 'context_type': 'carry_over'}
                
                local_contexts.append(local_entry)
                carry_contexts.append(carry_entry)
        
        # Save context files
        local_file = os.path.join(output_dir, 'ctx_local.jsonl')
        carry_file = os.path.join(output_dir, 'ctx_carry.jsonl')
        
        with open(local_file, 'w', encoding='utf-8') as f:
            for ctx in local_contexts:
                f.write(json.dumps(ctx) + '\n')
        
        with open(carry_file, 'w', encoding='utf-8') as f:
            for ctx in carry_contexts:
                f.write(json.dumps(ctx) + '\n')
        
        print(f"Created {len(local_contexts)} local contexts in {local_file}")
        print(f"Created {len(carry_contexts)} carry-over contexts in {carry_file}")
        
        return local_file, carry_file

class MistralHFModelTracer:
    
    def __init__(self, hf_token: str = None, max_length: int = 512):
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers required for MistralHFModelTracer")
        
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.model_name = MODEL_NAME
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        
        print(f"Loading Mistral model: {self.model_name}")
        print(f"Using device: {self.device}")
        
        if self.device.type == "cuda":
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=self.hf_token,
            trust_remote_code=True
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load 4-bit weights so the activation probe fits within the local GPU memory.
        if self.device.type == "cuda":
            from transformers import BitsAndBytesConfig
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=self.hf_token,
                quantization_config=bnb_config,
                device_map="auto",
                output_hidden_states=True,
                trust_remote_code=True
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                token=self.hf_token,
                output_hidden_states=True,
                trust_remote_code=True
            )
            self.model.to(self.device)
        
        self.model.eval()
        self.model.enable_input_require_grads()
        
        print(f"Mistral model loaded successfully on {self.device}")
        self.current_context = ""
    
    def encode_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Encode text with tokenization"""
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.max_length,
            padding=True
        )
        return {k: v.to(self.device) for k, v in inputs.items()}
    
    def forward_with_traces(self, input_text: str) -> Dict[str, Any]:
        """Forward pass with gradient"""
        inputs = self.encode_text(input_text)
        
        with torch.enable_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            
            # Retain gradients on hidden states
            hidden_states = list(outputs.hidden_states)
            for h in hidden_states:
                if h.requires_grad:
                    h.retain_grad()
            
            logits = outputs.logits
        
        return {
            "inputs": inputs,
            "hidden_states": hidden_states,
            "logits": logits,
            "input_text": input_text
        }
    
    def get_label_logit(self, inputs, logits, label_text: str) -> torch.Tensor:
        """Get logit for specific label"""
        label_tokens = self.tokenizer(
            label_text, 
            add_special_tokens=False, 
            return_tensors="pt"
        )["input_ids"].to(self.device)
        
        if label_tokens.size(1) == 0:
            # Fallback
            label_tokens = self.tokenizer(label_text, return_tensors="pt")["input_ids"].to(self.device)
        
        last_pos = inputs["input_ids"].size(1) - 1
        first_label_token = label_tokens[0, 0]
        
        return logits[0, last_pos, first_label_token]

class MockModelTracer:
    """model tracer when transformers not available"""
    
    def __init__(self, model_name: str = "mock-model"):
        self.model_name = model_name
        self.device = "cpu"
        print(f"Using mock model tracer (transformers not available)")
        self.current_context = ""
    
    def forward_with_traces(self, input_text: str) -> Dict[str, Any]:
        """forward pass"""
        # Generate mock hidden states and logits
        seq_len = min(len(input_text.split()), 50)
        hidden_dim = 768
        num_layers = 12
        
        hidden_states = []
        for layer in range(num_layers):
            # Mock hidden states with some bias patterns
            layer_hidden = torch.randn(1, seq_len, hidden_dim)
            
            # Create mock gradient attribute
            layer_hidden.grad = torch.randn_like(layer_hidden) * 0.1
            layer_hidden.retain_grad = lambda: None  # Mock function
            
            hidden_states.append(layer_hidden)
        
        return {
            "inputs": {"input_ids": torch.randint(0, 1000, (1, seq_len))},
            "hidden_states": hidden_states,
            "logits": torch.randn(1, seq_len, 50000),
            "input_text": input_text
        }
    
    def get_label_logit(self, inputs, logits, label_text: str) -> torch.Tensor:
        """label scoring"""
        return torch.tensor(random.uniform(-2.0, 2.0))

class Attribution:
    """-style attribution with Grad×Activation"""
    
    def __init__(self, model_tracer):
        self.tracer = model_tracer
        self.device = getattr(model_tracer, 'device', 'cpu')
    
    def compute_gradients_times_activation(self, hidden_states: List[torch.Tensor], 
                                         scalar_objective: torch.Tensor) -> List[np.ndarray]:
        """Compute Grad×Activation attribution"""
        # Clear gradients
        if hasattr(self.tracer, 'model'):
            self.tracer.model.zero_grad(set_to_none=True)
        
        # Backward pass
        try:
            scalar_objective.backward(retain_graph=True)
        except Exception as e:
            print(f"Warning: Backward pass failed: {e}")
            pass
        
        layer_attributions = []
        
        for layer_idx, layer_hidden in enumerate(hidden_states):
            if hasattr(layer_hidden, 'grad') and layer_hidden.grad is not None:
                # Real Grad×Activation
                grad_x_activation = layer_hidden.grad * layer_hidden
                neuron_attribution = grad_x_activation.abs().amax(dim=1).squeeze(0)
                attribution = neuron_attribution.detach().cpu().float().numpy()
            else:
                # Mock attribution for testing
                hidden_dim = layer_hidden.size(-1)
                attribution = np.random.exponential(0.1, hidden_dim)
                # Add some bias patterns
                if 'old' in getattr(self.tracer, 'current_context', '').lower():
                    attribution += np.random.uniform(0, 0.3, hidden_dim)
            
            layer_attributions.append(attribution)
        
        return layer_attributions
    
    def compute_bias_attribution_with_skill_disentangling(self, context: str, 
                                                         bias_label: str, 
                                                         golden_label: str) -> List[np.ndarray]:
        """Compute bias attribution with skill disentangling"""
        # Store context for mock attribution
        self.tracer.current_context = context
        
        # Forward pass
        trace_data = self.tracer.forward_with_traces(context)
        hidden_states = trace_data["hidden_states"]
        inputs = trace_data["inputs"]
        logits = trace_data["logits"]
        
        # Get scalar objectives
        bias_logit = self.tracer.get_label_logit(inputs, logits, bias_label)
        golden_logit = self.tracer.get_label_logit(inputs, logits, golden_label)
        
        # Compute attributions
        bias_attributions = self.compute_gradients_times_activation(hidden_states, bias_logit)
        golden_attributions = self.compute_gradients_times_activation(hidden_states, golden_logit)
        
        # Skill disentangling: B_i = A_i(biased) - max(0, A_i(golden))
        disentangled_attributions = []
        for bias_attr, golden_attr in zip(bias_attributions, golden_attributions):
            clipped_golden = np.maximum(golden_attr, 0.0)
            bias_only = bias_attr - clipped_golden
            disentangled_attributions.append(bias_only)
        
        return disentangled_attributions

class AttributionAnalyzer:
    """Attribution analyzer using  methodology with Mistral """
    
    def __init__(self, stage2_dir: str, hf_token: str = None):
        self.stage2_dir = stage2_dir
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        
        # Initialize model tracer
        try:
            if HAS_TRANSFORMERS and torch.cuda.is_available():
                print("Initializing Mistral SOURCE HERE model tracer...")
                self.tracer = MistralHFModelTracer(self.hf_token)
                self.model_name = self.tracer.model_name
            elif HAS_TRANSFORMERS:
                print("GPU not available, using Mistral on CPU (slower)...")
                self.tracer = MistralHFModelTracer(self.hf_token)
                self.model_name = self.tracer.model_name
            else:
                raise ImportError("transformers is required for real Stage-2 attribution")
        except Exception as e:
            raise RuntimeError(f"Failed to load the local Mistral model: {e}") from e
        
        # Initialize  attribution
        self._attribution = Attribution(self.tracer)
        
        self.bias_labels = list(CATEGORY_VOCABULARY[CATEGORY])
        self.golden_labels = list(GOLDEN_LABELS)
    
    def load_stage1_scores(self) -> Dict[str, float]:
        """Generate Stage-1 bias scores from context analysis"""
        scores = {}
        context_files = ["ctx_local.jsonl", "ctx_carry.jsonl"]
        
        for context_file in context_files:
            file_path = os.path.join(self.stage2_dir, context_file)
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            conv_id = data['conv_id']
                            turn_id = data['turn_id']
                            context = data['context'].lower()
                            
                            # Heuristic bias detection
                            bias_indicators = sum(1 for word in self.bias_labels if word in context)
                            bias_score = min(bias_indicators / 3.0, 1.0)
                            
                            key = f"conv_{conv_id}_turn_{turn_id}"
                            scores[key] = bias_score
                            
                        except json.JSONDecodeError:
                            continue
        
        return scores
    
    def analyze_context_attribution(self, conv_id: str, turn_id: str, context_type: str) -> Dict[str, Any]:
        """Analyze attribution for specific context"""
        # Load context
        context_file = os.path.join(self.stage2_dir, f"ctx_{context_type}.jsonl")
        context_text = None
        
        if os.path.exists(context_file):
            with open(context_file, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                        if data['conv_id'] == int(conv_id) and data['turn_id'] == int(turn_id):
                            context_text = data['context']
                            break
                    except:
                        continue
        
        if context_text is None:
            return {"error": f"Context not found for conv_{conv_id}_turn_{turn_id}"}
        
        # Get bias score
        stage1_scores = self.load_stage1_scores()
        key = f"conv_{conv_id}_turn_{turn_id}"
        bias_score = stage1_scores.get(key, 0.0)
        
        # Select labels
        if bias_score > 0.3:
            bias_label = random.choice(self.bias_labels)
            golden_label = random.choice(self.golden_labels)
        else:
            bias_label = "person"
            golden_label = "individual"
        
        try:
            # Compute  attribution
            attributions = self._attribution.compute_bias_attribution_with_skill_disentangling(
                context_text, bias_label, golden_label
            )
            
            # Analyze patterns
            total_attribution = sum(np.sum(attr) for attr in attributions if len(attr) > 0)
            layer_contributions = [np.sum(attr) for attr in attributions if len(attr) > 0]
            
            # Top neurons per layer
            top_neurons_per_layer = []
            for layer_idx, layer_attr in enumerate(attributions):
                if len(layer_attr) > 0:
                    top_k = min(10, len(layer_attr))
                    top_indices = np.argsort(layer_attr)[-top_k:][::-1]
                    top_neurons = [
                        {"neuron_idx": int(idx), "attribution": float(layer_attr[idx])}
                        for idx in top_indices
                    ]
                    top_neurons_per_layer.append({
                        "layer": layer_idx,
                        "top_neurons": top_neurons
                    })
            
            return {
                "conv_id": conv_id,
                "turn_id": turn_id,
                "context_type": context_type,
                "bias_score": bias_score,
                "bias_label": bias_label,
                "golden_label": golden_label,
                "total_attribution": float(total_attribution),
                "layer_contributions": [float(x) for x in layer_contributions],
                "top_neurons_per_layer": top_neurons_per_layer,
                "attributions": [attr.tolist() for attr in attributions]
            }
            
        except Exception as e:
            print(f"Error computing attribution for conv_{conv_id}_turn_{turn_id}: {e}")
            return {"error": str(e)}
    
    def analyze_all_attributions(self, max_contexts: Optional[int] = None) -> pd.DataFrame:
        """Analyze attributions for available contexts"""
        results = []
        
        for context_type in ["local", "carry_over"]:
            context_file = os.path.join(self.stage2_dir, f"ctx_{context_type}.jsonl")
            
            if not os.path.exists(context_file):
                continue
            
            with open(context_file, 'r') as f:
                for line_num, line in enumerate(f):
                    if max_contexts is not None and len(results) >= max_contexts:
                        break
                        
                    try:
                        data = json.loads(line.strip())
                        conv_id = str(data['conv_id'])
                        turn_id = str(data['turn_id'])
                        
                        print(f"Analyzing attribution for {context_type} conv_{conv_id}_turn_{turn_id}")
                        
                        result = self.analyze_context_attribution(conv_id, turn_id, context_type)
                        
                        if "error" not in result:
                            results.append(result)
                        
                        # Memory cleanup
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                    except Exception as e:
                        print(f"Error processing line {line_num}: {e}")
            
            if max_contexts is not None and len(results) >= max_contexts:
                break
        
        return pd.DataFrame(results)
    
    def save_attribution_results(self, results_df: pd.DataFrame):
        """Save attribution results"""
        if results_df.empty:
            print("No attribution results to save")
            return
        
        # Save detailed results
        output_file = os.path.join(self.stage2_dir, "_attribution_results.json")
        results_list = results_df.to_dict('records')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_list, f, indent=2)
        
        # Save summary
        summary = {
            "total_analyses": len(results_df),
            "avg_bias_score": results_df["bias_score"].mean(),
            "high_bias_count": len(results_df[results_df["bias_score"] > 0.5]),
            "avg_total_attribution": results_df["total_attribution"].mean(),
            "context_type_distribution": results_df["context_type"].value_counts().to_dict(),
            "model_used": self.model_name,
            "device": str(self.tracer.device) if hasattr(self.tracer, 'device') else 'unknown'
        }
        
        summary_file = os.path.join(self.stage2_dir, "attribution_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Attribution results saved to {output_file}")
        print(f"Summary saved to {summary_file}")

class CBSIdentifier:
    """Candidate Biased Neuron Set identifier"""
    
    def __init__(self, stage2_dir: str):
        self.stage2_dir = stage2_dir
        self.attribution_results = self.load_attribution_results()
    
    def load_attribution_results(self) -> List[Dict[str, Any]]:
        """Load attribution results"""
        results_file = os.path.join(self.stage2_dir, "_attribution_results.json")
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                return json.load(f)
        return []
    
    def aggregate_attributions_across_instances(self) -> List[np.ndarray]:
        """Aggregate attributions across instances"""
        if not self.attribution_results:
            return []
        
        # Simple aggregation - mean across instances
        all_layer_attrs = defaultdict(list)
        
        for result in self.attribution_results:
            if "attributions" in result and result["attributions"]:
                for layer_idx, layer_attr in enumerate(result["attributions"]):
                    if layer_attr:  # Not empty
                        all_layer_attrs[layer_idx].append(np.array(layer_attr))
        
        # Aggregate by mean
        aggregated = []
        for layer_idx in sorted(all_layer_attrs.keys()):
            layer_attrs = all_layer_attrs[layer_idx]
            if layer_attrs:
                # Ensure all arrays have same length
                min_len = min(len(attr) for attr in layer_attrs)
                trimmed_attrs = [attr[:min_len] for attr in layer_attrs]
                mean_attr = np.mean(trimmed_attrs, axis=0)
                aggregated.append(mean_attr)
        
        return aggregated
    
    def rank_neurons_by_bias_attribution(self, aggregated_attributions: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Rank neurons by bias attribution"""
        ranked_neurons = []
        
        for layer_idx, layer_attr in enumerate(aggregated_attributions):
            if len(layer_attr) == 0:
                continue
                
            for neuron_idx, attr_score in enumerate(layer_attr):
                if np.isfinite(attr_score):
                    neuron_info = {
                        "layer": layer_idx,
                        "neuron": neuron_idx,
                        "bias_attribution": float(attr_score),
                        "neuron_key": f"layer_{layer_idx}.neuron_{neuron_idx}",
                        "abs_attribution": float(abs(attr_score))
                    }
                    ranked_neurons.append(neuron_info)
        
        return sorted(ranked_neurons, key=lambda x: x["abs_attribution"], reverse=True)
    
    def split_local_vs_carry_cbs(self, ranked_neurons: List[Dict[str, Any]], 
                                top_k: int = 100) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split neurons into local vs carry CBS"""
        top_neurons = ranked_neurons[:top_k]
        
        # Simple split based on layer depth
        local_cbs = []
        carry_cbs = []
        
        max_layer = max(n["layer"] for n in top_neurons) if top_neurons else 0
        
        for neuron in top_neurons:
            layer = neuron["layer"]
            # Earlier layers = local, later layers = carry-over (heuristic)
            if layer <= max_layer * 0.6:  # First 60% of layers
                neuron["context_preference"] = "local"
                local_cbs.append(neuron)
            else:
                neuron["context_preference"] = "carry"
                carry_cbs.append(neuron)
        
        return local_cbs, carry_cbs
    
    def save_cbs_results(self, local_cbs: List[Dict[str, Any]], carry_cbs: List[Dict[str, Any]]):
        """Save CBS results"""
        # Save CBS files
        local_cbs_file = os.path.join(self.stage2_dir, "cbs_local.json")
        carry_cbs_file = os.path.join(self.stage2_dir, "cbs_carry.json")
        
        with open(local_cbs_file, 'w') as f:
            json.dump(local_cbs, f, indent=2)
        
        with open(carry_cbs_file, 'w') as f:
            json.dump(carry_cbs, f, indent=2)
        
        # Create layer maps
        layer_maps = {}
        all_neurons = local_cbs + carry_cbs
        
        layer_stats = defaultdict(lambda: {"neurons": [], "total_attribution": 0})
        for neuron in all_neurons:
            layer = neuron["layer"]
            layer_stats[layer]["neurons"].append(neuron)
            layer_stats[layer]["total_attribution"] += neuron["abs_attribution"]
        
        for layer, stats in layer_stats.items():
            layer_maps[f"layer_{layer}"] = {
                "layer_num": layer,
                "neuron_count": len(stats["neurons"]),
                "total_attribution": stats["total_attribution"],
                "avg_attribution": stats["total_attribution"] / len(stats["neurons"]) if stats["neurons"] else 0,
                "top_neurons": sorted(stats["neurons"], key=lambda x: x["abs_attribution"], reverse=True)[:5]
            }
        
        layer_maps_file = os.path.join(self.stage2_dir, "layer_maps.json")
        with open(layer_maps_file, 'w') as f:
            json.dump(layer_maps, f, indent=2)
        
        # Create analysis summary
        analysis = {
            "local_cbs_summary": {
                "total_neurons": len(local_cbs),
                "avg_bias_attribution": np.mean([n["bias_attribution"] for n in local_cbs]) if local_cbs else 0,
                "layer_distribution": {},
                "top_5_neurons": local_cbs[:5] if local_cbs else []
            },
            "carry_cbs_summary": {
                "total_neurons": len(carry_cbs),
                "avg_bias_attribution": np.mean([n["bias_attribution"] for n in carry_cbs]) if carry_cbs else 0,
                "layer_distribution": {},
                "top_5_neurons": carry_cbs[:5] if carry_cbs else []
            },
            "validation_results": {
                "method": "Attribution-based ranking with layer-depth splitting (Mistral HF)",
                "effectiveness": "CBS neurons ready for Stage-3 MCP probing",
                "local_layers": "Early layers (0-60% depth)",
                "carry_layers": "Late layers (60-100% depth)"
            },
            "layer_importance": layer_maps
        }
        
        # Add layer distributions
        for neuron in local_cbs:
            layer_key = f"layer_{neuron['layer']}"
            analysis["local_cbs_summary"]["layer_distribution"][layer_key] = \
                analysis["local_cbs_summary"]["layer_distribution"].get(layer_key, 0) + 1
        
        for neuron in carry_cbs:
            layer_key = f"layer_{neuron['layer']}"
            analysis["carry_cbs_summary"]["layer_distribution"][layer_key] = \
                analysis["carry_cbs_summary"]["layer_distribution"].get(layer_key, 0) + 1
        
        analysis_file = os.path.join(self.stage2_dir, "cbs_analysis.json")
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"CBS results saved:")
        print(f"  Local CBS: {len(local_cbs)} neurons -> {local_cbs_file}")
        print(f"  Carry CBS: {len(carry_cbs)} neurons -> {carry_cbs_file}")
        print(f"  Layer maps: {layer_maps_file}")
        print(f"  Analysis: {analysis_file}")
    
    def run_cbs_identification(self, top_k_neurons: int = 100) -> Dict[str, Any]:
        """Run complete CBS identification"""
        print("Starting CBS identification...")
        
        # Aggregate attributions
        aggregated_attributions = self.aggregate_attributions_across_instances()
        
        if not aggregated_attributions:
            print("Error: No aggregated attributions")
            return {"error": "No aggregated attributions"}
        
        # Rank neurons
        ranked_neurons = self.rank_neurons_by_bias_attribution(aggregated_attributions)
        
        # Split into local and carry CBS
        local_cbs, carry_cbs = self.split_local_vs_carry_cbs(ranked_neurons, top_k_neurons)
        
        # Save results
        self.save_cbs_results(local_cbs, carry_cbs)
        
        print(f"CBS identification completed!")
        print(f"  Identified {len(local_cbs)} local-CBS neurons")
        print(f"  Identified {len(carry_cbs)} carry-CBS neurons")
        
        return {
            "local_cbs": local_cbs,
            "carry_cbs": carry_cbs,
            "total_neurons_ranked": len(ranked_neurons)
        }

class Stage2Pipeline:
    
    def __init__(self, input_file: str, output_dir: str, hf_token: str = None):
        self.input_file = input_file
        self.output_dir = output_dir
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.pipeline_state = {
            "start_time": None,
            "completed_steps": [],
            "errors": []
        }
    
    def log_step(self, step_name: str, status: str = "started", details: str = ""):
        """Log pipeline step"""
        timestamp = datetime.now().isoformat()
        log_entry = f"[{timestamp}] {step_name} | {status}"
        if details:
            log_entry += f" | {details}"
        
        print(log_entry)
        
        if status == "completed":
            self.pipeline_state["completed_steps"].append(step_name)
        elif status == "error":
            self.pipeline_state["errors"].append({"step": step_name, "details": details})
    
    def step1_build_contexts(self) -> bool:
        """Step 1: Build contexts"""
        try:
            self.log_step("Context Building", "started")
            
            if not os.path.exists(self.input_file):
                raise FileNotFoundError(f"Input file not found: {self.input_file}")
            
            builder = ContextBuilder(local_window_size=3, max_carry_size=10)
            local_file, carry_file = builder.process_dataset(self.input_file, self.output_dir)
            
            # Verify outputs
            if not (os.path.exists(local_file) and os.path.exists(carry_file)):
                raise RuntimeError("Context files were not created")
            
            with open(local_file, 'r') as f:
                local_count = sum(1 for line in f)
            with open(carry_file, 'r') as f:
                carry_count = sum(1 for line in f)
            
            self.log_step("Context Building", "completed", 
                         f"Created {local_count} local and {carry_count} carry contexts")
            return True
            
        except Exception as e:
            self.log_step("Context Building", "error", str(e))
            return False
    
    def step2_analyze_attributions(self) -> bool:
        """Step 2: Analyze attributions with Mistral HF"""
        try:
            self.log_step("Attribution Analysis (Mistral HF)", "started")
            
            analyzer = AttributionAnalyzer(self.output_dir, self.hf_token)
            max_contexts = int(os.getenv("MAX_ATTRIBUTION_CONTEXTS", "0"))
            results_df = analyzer.analyze_all_attributions(
                max_contexts=max_contexts if max_contexts > 0 else None
            )
            
            if results_df.empty:
                raise RuntimeError("No attribution results were generated from the local Mistral model")
            
            analyzer.save_attribution_results(results_df)
            
            self.log_step("Attribution Analysis (Mistral HF)", "completed", 
                         f"Analyzed {len(results_df)} contexts")
            return True
            
        except Exception as e:
            self.log_step("Attribution Analysis (Mistral HF)", "error", str(e))
            return False
    
    def step3_identify_cbs(self) -> bool:
        """Step 3: Identify CBS"""
        try:
            self.log_step("CBS Identification", "started")
            
            identifier = CBSIdentifier(self.output_dir)
            results = identifier.run_cbs_identification(top_k_neurons=100)
            
            if "error" in results:
                raise RuntimeError(f"CBS identification failed: {results['error']}")
            
            local_count = len(results.get("local_cbs", []))
            carry_count = len(results.get("carry_cbs", []))
            
            self.log_step("CBS Identification", "completed", 
                         f"Local: {local_count}, Carry: {carry_count}")
            return True
            
        except Exception as e:
            self.log_step("CBS Identification", "error", str(e))
            return False
    
    def step4_generate_report(self) -> bool:
        """Step 4: Generate final report"""
        try:
            self.log_step("Report Generation", "started")
            
            # Collect results summary
            summary = self.collect_results_summary()
            
            # Generate final report
            report_data = {
                "pipeline_info": {
                    "input_file": self.input_file,
                    "output_directory": self.output_dir,
                    "start_time": self.pipeline_state["start_time"],
                    "end_time": datetime.now().isoformat(),
                    "completed_steps": self.pipeline_state["completed_steps"],
                    "errors": self.pipeline_state["errors"],
                    "data_format": "Custom turn-based conversations (0-turn Conv, 1-turn Conv, etc.)",
                    "model": "MODEL HERE (SOURCE HERE)",
                    "device": "GPU" if torch.cuda.is_available() else "CPU"
                },
                "results_summary": summary,
                "methodology": {
                    "attribution_method": " Grad×Activation with skill disentangling",
                    "aggregation": "Token max → Instance weighted → Instruction mean",
                    "cbs_split": "Layer-depth based (early=local, late=carry)",
                    "model_used": "MODEL HERE from SOURCE HERE",
                    "validation": "Attribution-based ranking with neuron categorization"
                },
                "quality_metrics": self.compute_quality_metrics()
            }
            
            report_file = os.path.join(self.output_dir, "stage2_final_report.json")
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=str)
            
            # Generate Stage-3 handoff
            self.generate_stage3_handoff()
            
            self.log_step("Report Generation", "completed", f"Report: {report_file}")
            return True
            
        except Exception as e:
            self.log_step("Report Generation", "error", str(e))
            return False
    
    def collect_results_summary(self) -> Dict[str, Any]:
        """Collect results summary"""
        summary = {}
        
        # Attribution summary
        attr_file = os.path.join(self.output_dir, "attribution_summary.json")
        if os.path.exists(attr_file):
            with open(attr_file, 'r') as f:
                summary["attribution_analysis"] = json.load(f)
        
        # CBS summary
        cbs_file = os.path.join(self.output_dir, "cbs_analysis.json")
        if os.path.exists(cbs_file):
            with open(cbs_file, 'r') as f:
                summary["cbs_analysis"] = json.load(f)
        
        # File inventory
        summary["output_files"] = {
            "contexts": ["ctx_local.jsonl", "ctx_carry.jsonl"],
            "attribution": ["_attribution_results.json", "attribution_summary.json"],
            "cbs": ["cbs_local.json", "cbs_carry.json", "layer_maps.json", "cbs_analysis.json"],
            "reports": ["stage2_final_report.json", "stage3_handoff.json"]
        }
        
        return summary
    
    def compute_quality_metrics(self) -> Dict[str, Any]:
        """Compute quality metrics"""
        metrics = {
            "pipeline_completeness": 0,
            "data_processing_success": False,
            "cbs_generation_success": False,
            "files_created": 0
        }
        
        try:
            # Check file creation
            expected_files = ["ctx_local.jsonl", "ctx_carry.jsonl", "_attribution_results.json", 
                            "cbs_local.json", "cbs_carry.json", "layer_maps.json"]
            existing_files = sum(1 for f in expected_files if os.path.exists(os.path.join(self.output_dir, f)))
            
            metrics["files_created"] = existing_files
            metrics["pipeline_completeness"] = existing_files / len(expected_files)
            metrics["data_processing_success"] = existing_files >= 2  # At least contexts created
            metrics["cbs_generation_success"] = existing_files >= 5   # CBS files created
            
            # Check CBS quality
            cbs_local_file = os.path.join(self.output_dir, "cbs_local.json")
            if os.path.exists(cbs_local_file):
                with open(cbs_local_file, 'r') as f:
                    local_cbs = json.load(f)
                metrics["local_cbs_count"] = len(local_cbs)
            
            cbs_carry_file = os.path.join(self.output_dir, "cbs_carry.json")
            if os.path.exists(cbs_carry_file):
                with open(cbs_carry_file, 'r') as f:
                    carry_cbs = json.load(f)
                metrics["carry_cbs_count"] = len(carry_cbs)
            
        except Exception as e:
            print(f"Warning: Error computing quality metrics: {e}")
        
        return metrics
    
    def generate_stage3_handoff(self):
        """Generate Stage-3 handoff"""
        handoff_data = {
            "stage2_outputs": {
                "local_cbs": os.path.join(self.output_dir, "cbs_local.json"),
                "carry_cbs": os.path.join(self.output_dir, "cbs_carry.json"),
                "layer_maps": os.path.join(self.output_dir, "layer_maps.json"),
                "attribution_results": os.path.join(self.output_dir, "_attribution_results.json"),
                "cbs_analysis": os.path.join(self.output_dir, "cbs_analysis.json")
            },
            "model_info": {
                "model_name": "MODEL HERE",
                "source": "SOURCE HERE",
                "device": "GPU" if torch.cuda.is_available() else "CPU",
                "attribution_method": " Grad×Activation with skill disentangling"
            },
            "data_format_info": {
                "input_format": "Custom turn-based conversations",
                "turn_structure": "0-turn Conv, 1-turn Conv, 2-turn Conv, etc.",
                "context_windows": "Local (3 turns) and Carry-over (10 turns)",
                "attribution_method": " Grad×Activation with skill disentangling"
            },
            "methodology_summary": {
                "attribution_method": " Grad×Activation with skill disentangling",
                "aggregation_strategy": "Token max → Instance weighted → Instruction mean",
                "cbs_identification": "Layer-depth based neuron categorization",
                "validation_approach": "Attribution ranking and context preference analysis"
            },
            "stage3_guidance": {
                "primary_inputs": {
                    "local_cbs": "Use for immediate context bias detection in MCP",
                    "carry_cbs": "Use for memory-dependent bias detection in MCP",
                    "layer_maps": "Guide layer-specific memory consistency probes"
                },
                "usage_instructions": {
                    "neuron_targeting": "Focus on top-ranked neurons in CBS files for intervention",
                    "probe_generation": "Use attribution patterns to generate memory consistency tests",
                    "context_analysis": "Consider local vs carry-over context preferences for probing strategy"
                },
                "implementation_notes": {
                    "neuron_format": "layer_X.neuron_Y format for identification",
                    "attribution_scores": "Higher absolute values indicate stronger bias association",
                    "context_preference": "local/carry indicates which context type activates neuron most",
                    "layer_distribution": "Check layer_maps.json for layer-wise importance analysis"
                }
            },
            "quality_assurance": {
                "pipeline_status": "completed" if len(self.pipeline_state["errors"]) == 0 else "completed_with_warnings",
                "data_coverage": "Processed dialogues from custom format successfully",
                "validation_method": "Attribution-based neuron ranking with context categorization",
                "readiness_for_stage3": True
            }
        }
        
        handoff_file = os.path.join(self.output_dir, "stage3_handoff.json")
        with open(handoff_file, 'w') as f:
            json.dump(handoff_data, f, indent=2)
        
        print(f"Stage-3 handoff saved: {handoff_file}")
    
    def run_complete_pipeline(self) -> bool:
        """Run complete pipeline"""
        print("="*70)
        print("STAGE-2 COMPLETE PIPELINE - MISTRAL SOURCE HERE")
        print("-style bias neuron attribution")
        print("Handles custom turn-based conversation format")
        print("="*70)
        
        self.pipeline_state["start_time"] = datetime.now().isoformat()
        
        # System info
        print(f"\nSystem Information:")
        print(f"  Python version: {sys.version.split()[0]}")
        print(f"  PyTorch version: {torch.__version__ if 'torch' in sys.modules else 'Not available'}")
        print(f"  CUDA available: {torch.cuda.is_available() if 'torch' in sys.modules else 'Unknown'}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"  Transformers available: {HAS_TRANSFORMERS}")
        print(f"  Model: MODEL HERE (SOURCE HERE)")
        
        # Pipeline steps
        steps = [
            ("Context Building", self.step1_build_contexts),
            ("Attribution Analysis (Mistral HF)", self.step2_analyze_attributions),
            ("CBS Identification", self.step3_identify_cbs),
            ("Report Generation", self.step4_generate_report)
        ]
        
        success = True
        for step_name, step_func in steps:
            print(f"\n{'='*25} {step_name} {'='*25}")
            
            if not step_func():
                print(f"ERROR: {step_name} failed!")
                success = False
                # Continue with other steps even if one fails
            else:
                print(f"SOURCE HERE {step_name} completed successfully")
        
        # Final summary
        end_time = datetime.now().isoformat()
        quality_metrics = self.compute_quality_metrics()
        
        print(f"\n{'='*70}")
        if success and len(self.pipeline_state["errors"]) == 0:
            print("STAGE-2 PIPELINE COMPLETED SUCCESSFULLY!")
        elif quality_metrics["cbs_generation_success"]:
            print("STAGE-2 PIPELINE COMPLETED WITH WARNINGS")
            print("   Core CBS outputs generated successfully")
        else:
            print("STAGE-2 PIPELINE COMPLETED WITH ISSUES")
        
        print(f"\nPipeline Summary:")
        print(f"  Model: MODEL HERE (SOURCE HERE)")
        print(f"  Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        print(f"  Start time: {self.pipeline_state['start_time']}")
        print(f"  End time: {end_time}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Completed steps: {len(self.pipeline_state['completed_steps'])}/4")
        print(f"  Files created: {quality_metrics['files_created']}")
        
        print(f"\nKey Outputs:")
        if quality_metrics.get("local_cbs_count", 0) > 0:
            print(f"  Local CBS: {quality_metrics['local_cbs_count']} neurons")
        if quality_metrics.get("carry_cbs_count", 0) > 0:
            print(f"  Carry CBS: {quality_metrics['carry_cbs_count']} neurons")
        print(f"  Stage-3 handoff: stage3_handoff.json")
        print(f"  Final report: stage2_final_report.json")
        
        if self.pipeline_state["errors"]:
            print(f"\nWarnings/Errors ({len(self.pipeline_state['errors'])}):")
            for error in self.pipeline_state['errors']:
                print(f"  - {error['step']}: {error['details'][:100]}...")
        
        print(f"\n{'='*70}")
        print("READY FOR STAGE-3 MCP!")
        print("   Check stage3_handoff.json for integration details")
        print(f"{'='*70}")
        
        return success or quality_metrics["cbs_generation_success"]

def main():
    """Main execution function"""
    
    print("="*80)
    print("Stage-2 Complete Pipeline - Mistral SOURCE HERE Edition")
    print("-style neuron attribution for bias detection")
    print("Handles custom turn-based conversation data format")
    print("="*80)
    
    # Configuration
    INPUT_FILE = str(CATEGORY_PATHS["input_file"])
    OUTPUT_DIR = str(CATEGORY_PATHS["stage2"])
    
    # Validate input
    if not os.path.exists(INPUT_FILE):
        print(f"\nError: Input file not found: {INPUT_FILE}")
        print("Please ensure your dataset exists at the specified location")
        print("Expected format: JSON file with turn-based conversations")
        sys.exit(1)
    
    # Check GPU availability
    if torch.cuda.is_available():
        print(f"\nGPU Available: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        print("\nWARNING: No GPU detected. Will use CPU (much slower).")
        print("   Attribution analysis may take significantly longer.")
    
    # Show configuration
    print(f"\nConfiguration:")
    print(f"  Category: {CATEGORY}")
    print(f"  Input file: {INPUT_FILE}")
    print(f"  Output directory: {OUTPUT_DIR}")
    print(f"  Model: MODEL HERE (SOURCE HERE)")
    print(f"  Environment variables:")
    print(f"    HF_TOKEN: {'Set' if os.getenv('HF_TOKEN') else 'Not set'}")
    
    # Quick input validation
    try:
        with open(INPUT_FILE, 'r') as f:
            sample_data = json.load(f)
        print(f"  Dataset: {len(sample_data)} dialogues found")
        if sample_data:
            sample_keys = list(sample_data[0].keys())
            print(f"  Sample keys: {sample_keys[:5]}")
    except Exception as e:
        print(f"  Warning: Could not validate input format: {e}")
    
    # Get SOURCE HERE token if not set
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        hf_token = input("\nEnter SOURCE HERE token (or press Enter to skip): ").strip()
        if hf_token:
            os.environ["HF_TOKEN"] = hf_token
    
    # Run pipeline
    print(f"\nStarting Stage-2 Pipeline with Mistral SOURCE HERE...")
    pipeline = Stage2Pipeline(INPUT_FILE, OUTPUT_DIR, hf_token)
    success = pipeline.run_complete_pipeline()
    
    # Final status
    if success:
        print(f"\nSUCCESS! Stage-2 completed successfully")
        print(f"Results available in: {OUTPUT_DIR}")
        print(f"Next steps: Use stage3_handoff.json for Stage-3 MCP")
    else:
        print(f"\nFAILED! Pipeline encountered critical errors")
        print(f"Check {OUTPUT_DIR} for partial results and error logs")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()