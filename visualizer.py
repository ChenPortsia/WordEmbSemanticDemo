import numpy as np
import matplotlib.pyplot as plt
from adjustText import adjust_text
from typing import List, Optional, Tuple
from gensim.models import KeyedVectors

class SemanticSpaceVisualizer:
    def __init__(self):
        self.operations = {
            'add': np.add,
            'subtract': np.subtract,
            'multiply': np.multiply,
            'divide': np.divide,
            'average': np.mean
        }

    @staticmethod
    def normalize(vector: np.ndarray) -> np.ndarray:
        """Normalize a vector."""
        return vector / np.linalg.norm(vector)

    def word_in_model(self, words: List[str], model: KeyedVectors) -> List[str]:
        """Check if words are in the model's vocabulary."""
        return [word for word in words if word in model]

    def calculate_axis(self, model: KeyedVectors, base: List[str], contra: List[str]) -> np.ndarray:
        """Calculate the axis of the semantic space."""
        base_vec = np.mean([model[word] for word in self.word_in_model(base, model)], axis=0)
        contra_vec = np.mean([model[word] for word in self.word_in_model(contra, model)], axis=0)
        return self.normalize(base_vec - contra_vec)

    def project_words(self, model: KeyedVectors, words: List[str], x_axis: np.ndarray, y_axis: np.ndarray) -> Tuple[List[float], List[float]]:
        """Project words onto the semantic space axes."""
        vectors = [model[word] for word in self.word_in_model(words, model)]
        x_coords = [np.dot(vec, x_axis) for vec in vectors]
        y_coords = [np.dot(vec, y_axis) for vec in vectors]
        return x_coords, y_coords

    def semantic_space_2d_representation(self, model: KeyedVectors, x_space: Tuple[List[str], List[str]], 
                                         y_space: Tuple[List[str], List[str]], groups: List[List[str]], 
                                         operation: Optional[str] = None, target_group: str = 'all', 
                                         extra_word: Optional[str] = None) -> Tuple[List[float], List[float]]:
        """Generate 2D representation of semantic space."""
        x_axis = self.calculate_axis(model, x_space[0], x_space[1])
        y_axis = self.calculate_axis(model, y_space[0], y_space[1])

        if operation and extra_word:
            op_func = self.operations[operation]
            extra_vec = model[extra_word]
            for i, group in enumerate(groups):
                if target_group in ['all', f'group_{i+1}']:
                    groups[i] = [op_func(extra_vec, model[word]) for word in group if word in model]

        x_coords, y_coords = [], []
        for group in groups:
            x, y = self.project_words(model, group, x_axis, y_axis)
            x_coords.extend(x)
            y_coords.extend(y)

        return x_coords, y_coords

    def plot_semantic_space_2d(self, x_coords: List[float], y_coords: List[float], groups: List[List[str]], 
                               x_space: Tuple[List[str], List[str]], y_space: Tuple[List[str], List[str]]) -> plt.Figure:
        """Plot the semantic space in 2D."""
        fig, ax = plt.subplots(figsize=(12, 10))
        colors = ['blue', 'red', 'green']
        
        for i, group in enumerate(groups):
            ax.scatter(x_coords[sum(len(g) for g in groups[:i]):sum(len(g) for g in groups[:i+1])], 
                        y_coords[sum(len(g) for g in groups[:i]):sum(len(g) for g in groups[:i+1])], 
                        color=colors[i], label=f'Group {i+1}')

        texts = []
        for i, group in enumerate(groups):
            for j, word in enumerate(group):
                texts.append(ax.text(x_coords[sum(len(g) for g in groups[:i])+j], 
                                      y_coords[sum(len(g) for g in groups[:i])+j], word, fontsize=10))

        adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5))

        ax.set_xlabel(f'{x_space[0]} <---> {x_space[1]}')
        ax.set_ylabel(f'{y_space[0]} <---> {y_space[1]}')
        ax.set_title('Semantic Space 2D Representation')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        plt.tight_layout()
        return fig