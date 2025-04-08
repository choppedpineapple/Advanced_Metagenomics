import pandas as pd
import numpy as np
from Bio import SeqIO
from Bio.SeqUtils import GC
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from Bio.Seq import Seq
import primer3
import random
from tqdm import tqdm

class MLPrimerDesigner:
    def __init__(self, model_path=None):
        """
        Initialize the ML-based primer designer.
        
        Parameters:
        -----------
        model_path : str, optional
            Path to a pre-trained model file. If not provided, a new model will be trained.
        """
        if model_path:
            self.model = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
        else:
            self.model = None
            print("No model loaded. Please train or load a model first.")
        
        # Default parameters for primer design
        self.primer_length_range = (18, 25)
        self.tm_range = (58.0, 62.0)
        self.gc_range = (40.0, 60.0)
        
    def _extract_features(self, sequences):
        """
        Extract features from DNA primer sequences.
        
        Parameters:
        -----------
        sequences : list
            List of DNA sequences as strings.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing extracted features.
        """
        features = []
        
        for seq in sequences:
            seq_obj = Seq(seq)
            
            # Basic sequence properties
            length = len(seq)
            gc_content = GC(seq)
            
            # Calculate Tm using primer3
            tm = primer3.calcTm(seq)
            
            # Count nucleotides
            a_count = seq.count('A')
            c_count = seq.count('C')
            g_count = seq.count('G')
            t_count = seq.count('T')
            
            # Calculate dinucleotide frequencies
            dinucleotides = {}
            for i in range(len(seq) - 1):
                dinuc = seq[i:i+2]
                dinucleotides[dinuc] = dinucleotides.get(dinuc, 0) + 1
                
            # Normalize dinucleotide counts by sequence length
            for dinuc in ['AA', 'AC', 'AG', 'AT', 'CA', 'CC', 'CG', 'CT', 'GA', 'GC', 'GG', 'GT', 'TA', 'TC', 'TG', 'TT']:
                dinucleotides[dinuc] = dinucleotides.get(dinuc, 0) / (length - 1)
            
            # Self-complementarity score (using primer3)
            hairpin_score = primer3.calcHairpin(seq).dg
            dimer_score = primer3.calcHomodimer(seq).dg
            
            # End stability (G/C at 3' end)
            end_stability = 1 if seq[-1] in ['G', 'C'] else 0
            
            # Calculate repeats
            max_homopolymer = 1
            current = 1
            for i in range(1, len(seq)):
                if seq[i] == seq[i-1]:
                    current += 1
                else:
                    max_homopolymer = max(max_homopolymer, current)
                    current = 1
            max_homopolymer = max(max_homopolymer, current)
            
            # Assemble feature dictionary
            feat_dict = {
                'length': length,
                'gc_content': gc_content,
                'tm': tm,
                'a_count': a_count / length,
                'c_count': c_count / length,
                'g_count': g_count / length,
                't_count': t_count / length,
                'hairpin_score': hairpin_score,
                'dimer_score': dimer_score,
                'end_stability': end_stability,
                'max_homopolymer': max_homopolymer
            }
            
            # Add dinucleotide frequencies
            feat_dict.update(dinucleotides)
            
            features.append(feat_dict)
            
        return pd.DataFrame(features)
    
    def train(self, good_primers, bad_primers):
        """
        Train the machine learning model using good and bad primer sequences.
        
        Parameters:
        -----------
        good_primers : list
            List of good/successful primer sequences.
        bad_primers : list
            List of bad/failed primer sequences.
            
        Returns:
        --------
        dict
            Dictionary containing model performance metrics.
        """
        print("Extracting features from primers...")
        
        # Extract features
        good_features = self._extract_features(good_primers)
        bad_features = self._extract_features(bad_primers)
        
        # Create labels
        good_labels = np.ones(len(good_primers))
        bad_labels = np.zeros(len(bad_primers))
        
        # Combine data
        X = pd.concat([good_features, bad_features], axis=0)
        y = np.concatenate([good_labels, bad_labels])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
        
        # Train Random Forest model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        print("Model performance:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
            
        # Get feature importance
        feature_imp = dict(zip(X.columns, self.model.feature_importances_))
        top_features = sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)[:10]
        
        print("\nTop 10 important features:")
        for feature, importance in top_features:
            print(f"  {feature}: {importance:.4f}")
            
        return metrics
    
    def save_model(self, filepath):
        """Save the trained model to a file."""
        if self.model is None:
            raise ValueError("No model has been trained yet.")
        
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
        
    def load_model(self, filepath):
        """Load a trained model from a file."""
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        
    def predict_primer_quality(self, primers):
        """
        Predict the quality of primers using the trained model.
        
        Parameters:
        -----------
        primers : list
            List of primer sequences to evaluate.
            
        Returns:
        --------
        numpy.ndarray
            Array of prediction probabilities for each primer.
        """
        if self.model is None:
            raise ValueError("No model has been trained yet.")
        
        features = self._extract_features(primers)
        return self.model.predict_proba(features)[:, 1]  # Probability of being a good primer
    
    def generate_candidate_primers(self, template_seq, target_region, n_candidates=100):
        """
        Generate candidate primers for a specific target region.
        
        Parameters:
        -----------
        template_seq : str
            The full template DNA sequence.
        target_region : tuple
            (start, end) positions of the target region (0-based).
        n_candidates : int
            Number of candidate primers to generate.
            
        Returns:
        --------
        dict
            Dictionary containing forward and reverse primer candidates.
        """
        start, end = target_region
        
        # Regions where primers can be designed
        forward_region = template_seq[max(0, start - 200):start]
        reverse_region = template_seq[end:min(len(template_seq), end + 200)]
        
        # Initialize containers for primer candidates
        forward_candidates = []
        reverse_candidates = []
        
        # Generate candidates for forward primers
        for _ in range(n_candidates):
            length = random.randint(self.primer_length_range[0], self.primer_length_range[1])
            
            if len(forward_region) < length:
                continue
                
            start_pos = random.randint(0, len(forward_region) - length)
            primer = forward_region[start_pos:start_pos + length]
            
            forward_candidates.append(primer)
        
        # Generate candidates for reverse primers
        for _ in range(n_candidates):
            length = random.randint(self.primer_length_range[0], self.primer_length_range[1])
            
            if len(reverse_region) < length:
                continue
                
            start_pos = random.randint(0, len(reverse_region) - length)
            primer = str(Seq(reverse_region[start_pos:start_pos + length]).reverse_complement())
            
            reverse_candidates.append(primer)
        
        return {
            'forward': forward_candidates,
            'reverse': reverse_candidates
        }
    
    def design_primers(self, template_seq, target_region, n_candidates=100, n_to_return=5):
        """
        Design primers for a specific target region using the ML model.
        
        Parameters:
        -----------
        template_seq : str
            The full template DNA sequence.
        target_region : tuple
            (start, end) positions of the target region (0-based).
        n_candidates : int
            Number of candidate primers to generate.
        n_to_return : int
            Number of top-scoring primer pairs to return.
            
        Returns:
        --------
        list
            List of dictionaries containing top primer pairs and their properties.
        """
        if self.model is None:
            raise ValueError("No model has been trained yet.")
        
        # Generate candidate primers
        candidates = self.generate_candidate_primers(
            template_seq, target_region, n_candidates
        )
        
        # Predict quality scores
        forward_scores = self.predict_primer_quality(candidates['forward'])
        reverse_scores = self.predict_primer_quality(candidates['reverse'])
        
        # Combine forward and reverse primers into pairs
        pairs = []
        
        for i, (f_primer, f_score) in enumerate(zip(candidates['forward'], forward_scores)):
            for j, (r_primer, r_score) in enumerate(zip(candidates['reverse'], reverse_scores)):
                # Calculate combined score
                combined_score = (f_score + r_score) / 2
                
                # Calculate additional pair properties
                product_size = target_region[1] - target_region[0] + len(f_primer) + len(r_primer)
                
                # Check for heterodimer formation between primers
                heterodimer_score = primer3.calcHeterodimer(f_primer, r_primer).dg
                
                # Only consider pairs with good heterodimer scores
                if heterodimer_score > -6.0:  # Less negative is better (less stable dimer)
                    pairs.append({
                        'forward_primer': f_primer,
                        'reverse_primer': r_primer,
                        'forward_score': f_score,
                        'reverse_score': r_score,
                        'combined_score': combined_score,
                        'product_size': product_size,
                        'heterodimer_score': heterodimer_score
                    })
        
        # Sort pairs by combined score
        pairs.sort(key=lambda x: x['combined_score'], reverse=True)
        
        # Return top pairs
        return pairs[:n_to_return]


# Example usage

def load_example_data():
    """
    Load example data for training the model.
    In a real scenario, this would load from files or databases.
    """
    # Example of good primers (these are simplified examples)
    good_primers = [
        "ACGTACGTACGTACGTAC",
        "GCTAGCTAGCTAGCTAG",
        "ATCGATCGATCGATCGAT",
        # Add more good primers here...
    ]
    
    # Example of bad primers
    bad_primers = [
        "AAAAAAAAAAAAAAAAA",
        "GCGCGCGCGCGCGCGCGC",
        "ATATATATATATATATAT",
        # Add more bad primers here...
    ]
    
    # Generate more synthetic examples
    good_primers.extend(_generate_synthetic_good_primers(100))
    bad_primers.extend(_generate_synthetic_bad_primers(100))
    
    return good_primers, bad_primers

def _generate_synthetic_good_primers(n):
    """Generate synthetic good primers."""
    bases = ['A', 'C', 'G', 'T']
    good_primers = []
    
    for _ in range(n):
        length = random.randint(18, 25)
        # Create a primer with balanced GC content
        primer = ''.join(random.choices(bases, weights=[0.25, 0.25, 0.25, 0.25], k=length))
        good_primers.append(primer)
    
    return good_primers

def _generate_synthetic_bad_primers(n):
    """Generate synthetic bad primers."""
    bad_primers = []
    
    for _ in range(n):
        primer_type = random.choice([
            'high_gc',            # High GC content
            'low_gc',             # Low GC content
            'homopolymer',        # Contains homopolymer
            'self_complementary', # Contains self-complementary regions
        ])
        
        if primer_type == 'high_gc':
            length = random.randint(18, 25)
            primer = ''.join(random.choices(['G', 'C'], k=length))
        
        elif primer_type == 'low_gc':
            length = random.randint(18, 25)
            primer = ''.join(random.choices(['A', 'T'], k=length))
        
        elif primer_type == 'homopolymer':
            base = random.choice(['A', 'C', 'G', 'T'])
            homopolymer = base * random.randint(6, 10)
            
            prefix = ''.join(random.choices(['A', 'C', 'G', 'T'], k=random.randint(5, 10)))
            suffix = ''.join(random.choices(['A', 'C', 'G', 'T'], k=random.randint(5, 10)))
            
            primer = prefix + homopolymer + suffix
        
        elif primer_type == 'self_complementary':
            # Create a hairpin structure
            stem_length = random.randint(4, 8)
            stem = ''.join(random.choices(['A', 'C', 'G', 'T'], k=stem_length))
            
            # Create reverse complement
            complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G'}
            rev_comp = ''.join(complement[base] for base in reversed(stem))
            
            # Create loop
            loop = ''.join(random.choices(['A', 'C', 'G', 'T'], k=random.randint(3, 5)))
            
            # Create flanking regions
            prefix = ''.join(random.choices(['A', 'C', 'G', 'T'], k=random.randint(3, 6)))
            suffix = ''.join(random.choices(['A', 'C', 'G', 'T'], k=random.randint(3, 6)))
            
            primer = prefix + stem + loop + rev_comp + suffix
        
        bad_primers.append(primer)
    
    return bad_primers

def example_pipeline():
    """Run an example of the primer design pipeline."""
    print("Loading example data...")
    good_primers, bad_primers = load_example_data()
    
    # Initialize and train the model
    print("Initializing ML primer designer...")
    designer = MLPrimerDesigner()
    
    print("Training the model...")
    metrics = designer.train(good_primers, bad_primers)
    
    # Save the trained model
    designer.save_model("primer_ml_model.pkl")
    
    # Example: Design primers for a target region
    print("\nDesigning primers for a sample sequence...")
    
    # Create a sample sequence
    template_seq = ''.join(random.choices(['A', 'C', 'G', 'T'], k=1000))
    
    # Define target region (e.g., positions 400-600)
    target_region = (400, 600)
    
    # Design primers
    primer_pairs = designer.design_primers(
        template_seq, target_region, n_candidates=100, n_to_return=5
    )
    
    # Print results
    print("\nTop primer pairs:")
    for i, pair in enumerate(primer_pairs):
        print(f"\nPrimer Pair {i+1}:")
        print(f"  Forward: {pair['forward_primer']}")
        print(f"  Forward Score: {pair['forward_score']:.4f}")
        print(f"  Reverse: {pair['reverse_primer']}")
        print(f"  Reverse Score: {pair['reverse_score']:.4f}")
        print(f"  Combined Score: {pair['combined_score']:.4f}")
        print(f"  Product Size: {pair['product_size']} bp")
        print(f"  Heterodimer Score: {pair['heterodimer_score']:.2f}")

if __name__ == "__main__":
    example_pipeline()
