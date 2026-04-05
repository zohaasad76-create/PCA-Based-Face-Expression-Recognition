import numpy as np
import os
from PIL import Image
import glob

class ExpressionPCA:
   
    def __init__(self, variance_threshold=0.99):
        self.variance_threshold = variance_threshold
        self.mean = None
        self.std = None
        self.pca_basis = None
        self.eigenvalues = None
        
    def fit(self, images):
        n_pixels, n_images = images.shape
        self.mean = np.mean(images, axis=1, keepdims=True)
        self.std = np.std(images, axis=1, keepdims=True, ddof=1) + 1e-8
        standardized = (images - self.mean) / self.std
        U, S, Vt = np.linalg.svd(standardized, full_matrices=False)
        eigenvalues = (S ** 2) / (n_images - 1)
        eigenvectors = U
        total_variance = np.sum(eigenvalues)
        cumsum_variance = np.cumsum(eigenvalues)
        n_components = np.searchsorted(cumsum_variance, self.variance_threshold * total_variance) + 1
        n_components = min(n_components, len(eigenvalues))
        self.eigenvalues = eigenvalues[:n_components]
        self.pca_basis = eigenvectors[:, :n_components]
        
        variance_explained = np.sum(self.eigenvalues) / total_variance * 100
        print(f"  Retained {n_components}/{len(eigenvalues)} components - {variance_explained:.2f}% variance")
        
    def reconstruct(self, test_image):
        standardized = (test_image - self.mean) / self.std
        coefficients = self.pca_basis.T @ standardized
        reconstructed_std = self.pca_basis @ coefficients
        reconstructed = reconstructed_std * self.std + self.mean
        loss = np.linalg.norm(test_image - reconstructed)
        return reconstructed, loss


def load_image_as_vector(image_path):
  
    img = Image.open(image_path).convert('L')
    img_array = np.array(img, dtype=np.float64)
    return img_array.flatten().reshape(-1, 1)


def load_yale_faces_by_expression(data_dir):

    expressions = ['happy', 'normal', 'sad', 'sleepy', 'surprised', 'wink']
    
    expression_images = {expr: [] for expr in expressions}
    all_files = sorted(glob.glob(os.path.join(data_dir, "subject*")))  
    for img_path in all_files:
        if not os.path.isfile(img_path) or img_path.endswith('.txt'):
            continue
        basename = os.path.basename(img_path)
        try:
            person_id = int(basename.split('.')[0].replace('subject', ''))
            basename_lower = basename.lower()
            for expr in expressions:
                if expr in basename_lower:
                    expression_images[expr].append((person_id, img_path))
                    break
        except ValueError:
            continue
    
    return expression_images


def split_by_subjects(expression_images, n_train_subjects=10, n_test_subjects=5):
    all_persons = set()
    for expr_list in expression_images.values():
        for person_id, _ in expr_list:
            all_persons.add(person_id)
    all_persons = sorted(list(all_persons))
    if len(all_persons) < n_train_subjects + n_test_subjects:
        print(f"Warning: Only {len(all_persons)} subjects found, need {n_train_subjects + n_test_subjects}")
        n_test_subjects = max(1, len(all_persons) - n_train_subjects)
    np.random.shuffle(all_persons)
    train_subjects = set(all_persons[:n_train_subjects])
    test_subjects = set(all_persons[n_train_subjects:n_train_subjects + n_test_subjects])
    
    print(f"   Train subjects ({len(train_subjects)}): {sorted(train_subjects)}")
    print(f"   Test subjects ({len(test_subjects)}): {sorted(test_subjects)}")
    train_dict = {expr: [] for expr in expression_images.keys()}
    test_dict = {expr: [] for expr in expression_images.keys()}
    
    for expr, img_list in expression_images.items():
        for person_id, img_path in img_list:
            if person_id in train_subjects:
                train_dict[expr].append(img_path)
            elif person_id in test_subjects:
                test_dict[expr].append(img_path)
    
    return train_dict, test_dict, train_subjects, test_subjects


def main():
    DATA_DIR = "yalefaces"
    N_TRAIN_SUBJECTS = 10
    N_TEST_SUBJECTS = 5
    VARIANCE_THRESHOLD = 0.99
    SAVE_DIR = "saved_models_expression"
    ALL_EXPRESSIONS = ['happy', 'normal', 'sad', 'sleepy', 'surprised', 'wink']
    print("=" * 60)
    print("Task 2: PCA-Based Facial Expression Recognition")
    print("=" * 60)
    
    np.random.seed(42)
    os.makedirs(SAVE_DIR, exist_ok=True)
    print("\n1. Loading YALE Face Database by expression...")
    expression_images = load_yale_faces_by_expression(DATA_DIR)
    print(f"\n   Expression distribution:")
    for expr in ALL_EXPRESSIONS:
        print(f"   {expr.capitalize():10s}: {len(expression_images[expr])} images")
    print(f"\n2. Splitting subjects ({N_TRAIN_SUBJECTS} train, {N_TEST_SUBJECTS} test)...")
    train_dict, test_dict, train_subjects, test_subjects = split_by_subjects(
        expression_images, N_TRAIN_SUBJECTS, N_TEST_SUBJECTS
    )
    
    print(f"\n   Training images per expression:")
    for expr in ALL_EXPRESSIONS:
        print(f"   {expr.capitalize():10s}: {len(train_dict[expr])} images")
    
    print(f"\n   Test images per expression:")
    for expr in ALL_EXPRESSIONS:
        print(f"   {expr.capitalize():10s}: {len(test_dict[expr])} images")
    print("\n3. Training PCA for each expression...")
    pca_models = {}
    
    for expr in ALL_EXPRESSIONS:
        if len(train_dict[expr]) == 0:
            print(f"\n{expr.capitalize()}: No training images, skipping...")
            continue
            
        print(f"\n{expr.capitalize()}:")
        train_images = []
        for img_path in train_dict[expr]:
            img_vec = load_image_as_vector(img_path)
            train_images.append(img_vec)
        
        train_matrix = np.hstack(train_images)
        print(f"  Training matrix shape: {train_matrix.shape}")
        pca = ExpressionPCA(variance_threshold=VARIANCE_THRESHOLD)
        pca.fit(train_matrix)
        pca_models[expr] = pca
        save_path = os.path.join(SAVE_DIR, f"expression_{expr}.npz")
        np.savez(save_path,
                 mean=pca.mean,
                 std=pca.std,
                 pca_basis=pca.pca_basis,
                 eigenvalues=pca.eigenvalues)
        print(f"  Saved to {save_path}")
    
    print(f"\n✓ All PCA models saved to '{SAVE_DIR}/' directory")
    print("\n" + "=" * 60)
    print("4. Testing (Expression Recognition)")
    print("=" * 60)
    
    correct = 0
    total = 0
    confusion_matrix = {e: {ee: 0 for ee in ALL_EXPRESSIONS} for e in ALL_EXPRESSIONS}
    
    for true_expr in ALL_EXPRESSIONS:
        if true_expr not in pca_models:
            continue
            
        for test_img_path in test_dict[true_expr]:
            test_image = load_image_as_vector(test_img_path)
            losses = {}
            for expr, pca in pca_models.items():
                _, loss = pca.reconstruct(test_image)
                losses[expr] = loss
            predicted_expr = min(losses, key=losses.get)
            total += 1
            if predicted_expr == true_expr:
                correct += 1
                result = "✓"
            else:
                result = "✗"
            
            confusion_matrix[true_expr][predicted_expr] += 1
            
            test_filename = os.path.basename(test_img_path)
            print(f"{result} True: {true_expr:10s} | Predicted: {predicted_expr:10s} | "
                  f"Loss: {losses[predicted_expr]:8.2f} | {test_filename}")
    
    accuracy = 100 * correct / total if total > 0 else 0
    print("\n" + "=" * 60)
    print(f"RESULTS: {correct}/{total} correct")
    print(f"Accuracy: {accuracy:.2f}%")
    print("=" * 60)
    
    if total > 0:
        print("\nConfusion Matrix:")
        print("(Rows = True, Columns = Predicted)")
        print()
        
        print("             ", end="")
        for expr in ALL_EXPRESSIONS:
            print(f"{expr[:6]:>8s}", end="")
        print()
        
        for true_expr in ALL_EXPRESSIONS:
            print(f"{true_expr:12s} ", end="")
            for pred_expr in ALL_EXPRESSIONS:
                count = confusion_matrix[true_expr][pred_expr]
                print(f"{count:8d}", end="")
            print()


if __name__ == "__main__":
    main()
