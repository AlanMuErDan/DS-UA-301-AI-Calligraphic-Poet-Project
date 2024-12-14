import os
from utils.OCR_detector import ocr_single_character

def evaluate_ocr_accuracy(folder1, folder2):
    """
    Compare OCR results from two folders and calculate OCR accuracy.
    
    Args:
        folder1 (str): Path to the first folder (e.g., ground truth images).
        folder2 (str): Path to the second folder (e.g., generated images).
        
    Returns:
        float: OCR accuracy as a percentage.
    """
    # Get OCR results for both folders
    print("Performing OCR on the first folder...")
    ocr_results_1 = ocr_single_character(folder1)
    
    print("Performing OCR on the second folder...")
    ocr_results_2 = ocr_single_character(folder2)
    
    # Ensure both folders have the same number of images
    if len(ocr_results_1) != len(ocr_results_2):
        raise ValueError("Folders do not contain the same number of images!")
    
    # Calculate accuracy
    correct_count = sum(1 for x, y in zip(ocr_results_1, ocr_results_2) if x == y)
    total_count = len(ocr_results_1)
    accuracy = (correct_count / total_count) * 100 if total_count > 0 else 0.0
    
    print(f"OCR Accuracy: {accuracy:.2f}% ({correct_count}/{total_count} correct matches)")
    return accuracy

if __name__ == "__main__":
    # Example usage
    # ---------------------------------------
    # Specify your folders here:
    # ---------------------------------------
    generated_images_path = "/gpfsnyu/scratch/yl10337/GAN_processed_bdsr"
    real_images_path = "/gpfsnyu/scratch/yl10337/bdsr/1000"
    
    accuracy = evaluate_ocr_accuracy(generated_images_path, real_images_path)
    print(f"Final OCR Accuracy: {accuracy:.2f}%")