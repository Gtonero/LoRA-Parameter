import torch
import json
import argparse

try:
    from safetensors import safe_open
    def load_state_dict(file_path):
        with safe_open(file_path, framework="pt", device="cpu") as f:
            return {key: f.get_tensor(key) for key in f.keys()}       
except ImportError:
    from safetensors.torch import load_file
    def load_state_dict(file_path):
        return load_file(file_path)

def count_parameters(state_dict, search_terms):
    relevant_params = {k: v for k, v in state_dict.items() if any(term in k for term in search_terms)}
    return sum(param.numel() for param in relevant_params.values()), relevant_params

def format_parameters(param_count):
    if param_count == 0:
        return "Not Detect"
    if param_count >= 1_000_000:
        formatted = f"{param_count / 1_000_000:.2f}M parameters"
    elif param_count >= 1_000:
        formatted = f"{param_count / 1_000:.2f}K parameters"
    else:
        formatted = f"{param_count} parameters"

    full_count_with_commas = f"{param_count:,}"
    return f"{formatted} ({full_count_with_commas})"

def get_weight_vector_and_average_by_block(state_dict, base_names, block_ranges):
    block_averages_and_max = {}

    for base_name, block_range in zip(base_names, block_ranges):
        for i in range(block_range):
            block_name = f"{base_name}_{i}"
            block_weights = []

            for k, v in state_dict.items():
                if block_name in k and "alpha" not in k and isinstance(v, torch.Tensor):
                    block_weights.append(v.to(torch.float32).flatten())

            if block_weights:
                combined_tensor = torch.cat(block_weights)
                avg_weight = combined_tensor.abs().mean().item() 
                max_weight = combined_tensor.abs().max().item()  
                block_averages_and_max[block_name] = (avg_weight, max_weight)
            else:
                block_averages_and_max[block_name] = ("Not Detect", "Not Detect")

    return block_averages_and_max

def main():
    parser = argparse.ArgumentParser(description="Calculate parameters of LoRA components.")
    parser.add_argument("-i", "--input", required=True, help="Path to the .safetensors file.")
    parser.add_argument("--debug_unet", action="store_true", help="Enable debugging mode for UNet keys.")
    parser.add_argument("--debug_para", action="store_true", help="debug parameter.")
    parser.add_argument("--debug", action="store_true", help="debug.")
    args = parser.parse_args()

    lora_model_path = args.input
    state_dict = load_state_dict(lora_model_path)
    

    
    
    unet_params, _ = count_parameters(state_dict, ['lora_unet'])
    conv_params, _ = count_parameters(state_dict, ['conv'])
    unet_single_params, _ = count_parameters(state_dict, ['lora_unet_single', 'single_layers'])
    unet_double_params, _ = count_parameters(state_dict, ['lora_unet_double', 'double_layers'])
    text_encoder_1_params, _ = count_parameters(state_dict, ['lora_te_text_model_encoder', 'lora_te1_text_model_encoder'])
    text_encoder_2_params, _ = count_parameters(state_dict, ['lora_te2'])
    text_encoder_3_params, _ = count_parameters(state_dict, ['lora_te3'])
    print(f"UNet                    : {format_parameters(unet_params)}")
    print(f"Conv layer UNet         : {format_parameters(conv_params)}")
    print(f"UNet single block [Flux]: {format_parameters(unet_single_params)}")
    print(f"UNet double block [Flux]: {format_parameters(unet_double_params)}")
    print(f"Text-Encoder 1 Clip_L   : {format_parameters(text_encoder_1_params)}")
    print(f"Text-Encoder 2 Clip_G   : {format_parameters(text_encoder_2_params)}")
    print(f"Text-Encoder 3 T5       : {format_parameters(text_encoder_3_params)}")


    unet_base_names = [
        "lora_unet_input_blocks",
        "lora_unet_middle_block",
        "lora_unet_output_blocks"
    ]
    unet_block_ranges = [9, 3, 9]
    
    unet_flux_name = [ 
        "lora_unet_single_blocks",
        "lora_unet_double_blocks",
    ]
    unet_flux_ranges = [38, 19]

    if args.debug:
        def debug_metadata(file_path):
            with safe_open(file_path, framework="pt", device="cpu") as f:
                metadata = f.metadata()
                print("\n[DEBUG] Metadata Content:")
                for key, value in metadata.items():
                    print(f"{key}: {value}")
        debug_metadata(args.input)
            
        def extract_metadata(file_path, target_keys):
            with safe_open(file_path, framework="pt", device="cpu") as f:
                metadata = f.metadata()
                results = {}

                if metadata:
                    for key in target_keys:
                        results[key] = metadata.get(key, "Not Found")
            
                dtype_found = None
                for key in f.keys():
                    if "weight" in key:
                        dtype_found = f.get_tensor(key).dtype
                        break  

                results["dtype"] = str(dtype_found) if dtype_found else "Not Found"
                return results
           
    target_keys = ["ss_network_dim", "ss_network_alpha", "dtype"]
    if args.debug:
        results = extract_metadata(lora_model_path, target_keys)
    if args.debug:
        if results:
            print(f"\nNetwork_DIM = {results['ss_network_dim']} Network_ALPHA = {results['ss_network_alpha']} dtype = {results['dtype']}")
        else:
            print("Metadata not found.")

    if any(name in key for key in state_dict.keys() for name in unet_base_names):
        unet_block_averages_and_max = get_weight_vector_and_average_by_block(state_dict, unet_base_names, unet_block_ranges)
        print("\nUNet block averages and max weights:")
        for block, (avg, max_val) in unet_block_averages_and_max.items():
            if isinstance(avg, str) or isinstance(max_val, str):
                print(f"{block} average weight: {avg}, max weight: {max_val}")
            else:
                print(f"{block} average weight: {avg:.16f}, max weight: {max_val:.16f}")

    if any(name in key for key in state_dict.keys() for name in unet_flux_name):
        unet_fluxblock_averages = get_weight_vector_and_average_by_block(state_dict, unet_flux_name, unet_flux_ranges)
        print("\nUNet Flux block averages:")
        for block, avg in unet_fluxblock_averages.items():
            if isinstance(avg, str):
                print(f"{block} average weight: {avg}")
            else:
                print(f"{block} average weight: {avg:.16f}")

    if any("lora_te1_text_model_encoder_layers" in key for key in state_dict.keys()):
        text_encoder_te1_layer_averages_and_max = get_weight_vector_and_average_by_block(state_dict, ["lora_te1_text_model_encoder_layers"], [12])
        print("\nText-Encoder TE1 layers average weights and parameters:")
        for layer, (avg, max_val) in text_encoder_te1_layer_averages_and_max.items():
            relevant_params = {k: v for k, v in state_dict.items() if layer in k and isinstance(v, torch.Tensor)}
            num_params = sum(param.numel() for param in relevant_params.values())
            formatted_params = f"{num_params:,}"
            short_layer_name = layer.replace("lora_te1_text_model_encoder_", "lora_te2_")
            if isinstance(avg, str):
                print(f"{short_layer_name} average weight: {avg}, max weight: {max_val}, parameters: {formatted_params}")
            else:
                print(f"{short_layer_name} average weight: {avg:.16f}, max weight: {max_val:.16f}, parameters: {formatted_params}")

    if any("lora_te2_text_model_encoder_layers" in key for key in state_dict.keys()):
        text_encoder_te2_layer_averages_and_max = get_weight_vector_and_average_by_block(state_dict, ["lora_te2_text_model_encoder_layers"], [32])
        print("\nText-Encoder TE2 layers average weights and parameters:")
        for layer, (avg, max_val) in text_encoder_te2_layer_averages_and_max.items():
            relevant_params = {k: v for k, v in state_dict.items() if layer in k and isinstance(v, torch.Tensor)}
            num_params = sum(param.numel() for param in relevant_params.values())
            formatted_params = f"{num_params:,}"
            short_layer_name = layer.replace("lora_te2_text_model_encoder_", "lora_te2_")
            if isinstance(avg, str):
                print(f"{short_layer_name} average weight: {avg}, max weight: {max_val}, parameters: {formatted_params}")
            else:
                print(f"{short_layer_name} average weight: {avg:.16f}, max weight: {max_val:.16f}, parameters: {formatted_params}")


    if any("lora_te3_text_model_encoder_layers" in key for key in state_dict.keys()):
        text_encoder_te3_layer_averages_and_max = get_weight_vector_and_average_by_block(state_dict, ["lora_te3_text_model_encoder_layers"], [24])
        print("\nText-Encoder TE3 layers average weights:")
        for layer, (avg, _) in text_encoder_te3_layer_averages_and_max.items(): 
            short_layer_name = layer.replace("lora_te3_text_model_encoder_", "lora_te3_")
            if isinstance(avg, str):
                print(f"{short_layer_name} average weight: {avg}")
            else:
                print(f"{short_layer_name} average weight: {avg:.16f}")
                
    if args.debug_unet:
        print("\n[DEBUG] UNet Keys and Weights:")
        for key, value in state_dict.items():
            if "lora_unet" in key and isinstance(value, torch.Tensor):
                num_parameters = value.numel()  
                avg_weight = value.abs().mean().item()  
                max_weight = value.abs().max().item()  
                print(f"{key} -> Parameters: {num_parameters}, Avg Weight: {avg_weight:.4f}, Max Weight: {max_weight:.4f}")
    
    if args.debug_para:    
        for key, value in state_dict.items():
            print(f"{key}: {value.shape}")
                
                
if __name__ == "__main__":
    main()
