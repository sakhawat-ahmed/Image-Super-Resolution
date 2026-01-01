import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

def analyze_results():
    """Analyze and visualize results"""
    
    results_dir = Path("../results")
    
    # Find JSON results files
    json_files = list(results_dir.glob("results_*.json"))
    
    if not json_files:
        print("No results files found in ../results/")
        return
    
    # Load the most recent results
    latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
    
    print(f"Analyzing results from: {latest_file.name}")
    
    with open(latest_file, 'r') as f:
        data = json.load(f)
    
    # Display basic information
    print("\n" + "="*60)
    print("RESULTS ANALYSIS")
    print("="*60)
    
    print(f"\nDate: {data['timestamp']}")
    print(f"Scale Factor: {data['config']['scale_factor']}")
    print(f"Device: {data['device']}")
    print(f"Batch Size: {data['config']['batch_size']}")
    print(f"Epochs: {data['config']['srcnn_epochs']}")
    
    print("\nModel Performance:")
    print("-" * 40)
    
    if 'results' in data and data['results']:
        models = list(data['results'].keys())
        psnr_values = []
        ssim_values = []
        
        for model, metrics in data['results'].items():
            psnr = metrics['psnr']
            ssim = metrics['ssim']
            psnr_values.append(psnr)
            ssim_values.append(ssim)
            
            print(f"\n{model}:")
            print(f"  PSNR: {psnr:.2f} ± {metrics['std_psnr']:.2f} dB")
            print(f"  SSIM: {ssim:.4f} ± {metrics['std_ssim']:.4f}")
        
        # Create comparison chart
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        x = np.arange(len(models))
        width = 0.35
        
        # PSNR comparison
        bars1 = axes[0].bar(x, psnr_values, width, label='PSNR', color='skyblue', alpha=0.7)
        axes[0].set_xlabel('Model')
        axes[0].set_ylabel('PSNR (dB)')
        axes[0].set_title('PSNR Comparison')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars1, psnr_values):
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{value:.2f}', ha='center', va='bottom')
        
        # SSIM comparison
        bars2 = axes[1].bar(x, ssim_values, width, label='SSIM', color='lightcoral', alpha=0.7)
        axes[1].set_xlabel('Model')
        axes[1].set_ylabel('SSIM')
        axes[1].set_title('SSIM Comparison')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(models)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars2, ssim_values):
            height = bar.get_height()
            axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.002,
                        f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(results_dir / 'performance_comparison.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\nComparison chart saved to: {results_dir / 'performance_comparison.png'}")
        
        # Determine best model
        best_model = max(data['results'].items(), key=lambda x: x[1]['psnr'])
        print(f"\nBest Model: {best_model[0]} with PSNR={best_model[1]['psnr']:.2f} dB")
        
    else:
        print("No results data available")
    
    print("\n" + "="*60)
    print("Analysis complete!")

def main():
    try:
        analyze_results()
    except Exception as e:
        print(f"Error analyzing results: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())