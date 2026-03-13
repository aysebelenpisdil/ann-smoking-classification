"""
Artificial Neural Network - Smoking Status Classification GUI
"""

import matplotlib
# Force Tkinter backend for macOS compatibility
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')


class ArtificialNeuralNetwork:
    """Feedforward Artificial Neural Network Implementation"""

    def __init__(self, layer_sizes, learning_rate=0.1, iterations=100, random_state=42):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.random_state = random_state
        self.weights = []
        self.biases = []
        self.loss_history = []

    def sigmoid(self, x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def mse_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def initialize_weights(self):
        np.random.seed(self.random_state)
        self.weights = []
        self.biases = []

        for i in range(len(self.layer_sizes) - 1):
            w = np.random.randn(
                self.layer_sizes[i], self.layer_sizes[i + 1]
            ) * 0.5
            b = np.zeros((1, self.layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

    def forward_propagation(self, X):
        self.activations = [X]

        for i in range(len(self.weights)):
            z = (
                np.dot(self.activations[-1], self.weights[i])
                + self.biases[i]
            )
            a = self.sigmoid(z)
            self.activations.append(a)

        return self.activations[-1]

    def backward_propagation(self, X, y):
        m = X.shape[0]

        error = self.activations[-1] - y
        deltas = [
            error * self.sigmoid_derivative(self.activations[-1])
        ]

        for i in range(len(self.weights) - 1, 0, -1):
            error = deltas[0].dot(self.weights[i].T)
            delta = error * self.sigmoid_derivative(self.activations[i])
            deltas.insert(0, delta)

        for i in range(len(self.weights)):
            self.weights[i] -= (
                self.learning_rate
                * self.activations[i].T.dot(deltas[i])
                / m
            )
            self.biases[i] -= (
                self.learning_rate
                * np.sum(deltas[i], axis=0, keepdims=True)
                / m
            )

    def fit(self, X, y, verbose=False):
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        self.initialize_weights()
        self.loss_history = []

        for epoch in range(self.iterations):
            y_pred = self.forward_propagation(X)
            loss = self.mse_loss(y, y_pred)
            self.loss_history.append(loss)
            self.backward_propagation(X, y)

        return self

    def predict(self, X):
        probability = self.forward_propagation(X)
        return (probability >= 0.5).astype(int).flatten()


class SmokingClassifierGUI:
    """GUI for Smoking Status Classification using ANN"""

    def __init__(self, root):
        self.root = root
        self.root.title("ANN - Smoking Status Classification")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')

        # Data variables
        self.X = None
        self.y = None
        self.X_scaled = None
        self.scaler = StandardScaler()
        self.data_loaded = False

        # Model parameters (optimal values from Grid Search)
        self.hidden1 = tk.IntVar(value=128)
        self.hidden2 = tk.IntVar(value=64)
        self.learning_rate = tk.DoubleVar(value=0.5)
        self.iterations = tk.IntVar(value=150)

        self.create_widgets()

    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="ANN - Smoking Status Classification",
            font=('Helvetica', 16, 'bold')
        )
        title_label.pack(pady=10)

        # Top panel - Data and Parameters
        top_frame = ttk.Frame(main_frame)
        top_frame.pack(fill=tk.X, pady=5)

        # Left panel - Data Loading
        data_frame = ttk.LabelFrame(top_frame, text="Dataset", padding="10")
        data_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.load_btn = ttk.Button(
            data_frame, text="Load Dataset", command=self.load_data
        )
        self.load_btn.pack(pady=5)

        self.data_label = ttk.Label(
            data_frame, text="Data not loaded", foreground='red'
        )
        self.data_label.pack(pady=5)

        # Right panel - Model Parameters
        param_frame = ttk.LabelFrame(
            top_frame, text="Model Parameters", padding="10"
        )
        param_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)

        # Hidden layer 1
        ttk.Label(param_frame, text="Hidden Layer 1:").grid(
            row=0, column=0, sticky=tk.W, pady=2
        )
        ttk.Entry(param_frame, textvariable=self.hidden1, width=10).grid(
            row=0, column=1, pady=2
        )

        # Hidden layer 2
        ttk.Label(param_frame, text="Hidden Layer 2:").grid(
            row=1, column=0, sticky=tk.W, pady=2
        )
        ttk.Entry(param_frame, textvariable=self.hidden2, width=10).grid(
            row=1, column=1, pady=2
        )

        # Learning rate
        ttk.Label(param_frame, text="Learning Rate:").grid(
            row=2, column=0, sticky=tk.W, pady=2
        )
        ttk.Entry(param_frame, textvariable=self.learning_rate, width=10).grid(
            row=2, column=1, pady=2
        )

        # Iterations
        ttk.Label(param_frame, text="Iterations:").grid(
            row=3, column=0, sticky=tk.W, pady=2
        )
        ttk.Entry(param_frame, textvariable=self.iterations, width=10).grid(
            row=3, column=1, pady=2
        )

        # Topology display
        self.topo_label = ttk.Label(
            param_frame,
            text="Topology: 25 -> 128 -> 64 -> 1",
            font=('Helvetica', 10, 'bold')
        )
        self.topo_label.grid(row=4, column=0, columnspan=2, pady=10)

        # Middle panel - Scenario Selection
        scenario_frame = ttk.LabelFrame(
            main_frame, text="Scenario Selection", padding="10"
        )
        scenario_frame.pack(fill=tk.X, pady=10)

        self.scenario_var = tk.IntVar(value=1)

        scenarios = [
            ("Scenario 1: Training = Test", 1),
            ("Scenario 2: 5-Fold Cross Validation", 2),
            ("Scenario 3: 10-Fold Cross Validation", 3),
            ("Scenario 4: 75-25 Split (5 different seeds)", 4)
        ]

        for text, value in scenarios:
            ttk.Radiobutton(
                scenario_frame, text=text, variable=self.scenario_var,
                value=value
            ).pack(anchor=tk.W, pady=2)

        # Run button
        self.run_btn = ttk.Button(
            scenario_frame,
            text="Train and Test Model",
            command=self.run_scenario,
            style='Accent.TButton'
        )
        self.run_btn.pack(pady=10)

        # Bottom panel - Results
        result_frame = ttk.LabelFrame(main_frame, text="Results", padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Left - Text results
        text_frame = ttk.Frame(result_frame)
        text_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.result_text = tk.Text(
            text_frame, height=15, width=45, font=('Courier', 10)
        )
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Right - Graph
        self.graph_frame = ttk.Frame(result_frame)
        self.graph_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Network visualization button
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=5)

        ttk.Button(
            btn_frame, text="Show Network Architecture",
            command=self.show_network
        ).pack(side=tk.LEFT, padx=5)
        ttk.Button(
            btn_frame, text="Compare All Scenarios",
            command=self.compare_all
        ).pack(side=tk.LEFT, padx=5)

    def load_data(self):
        try:
            df = pd.read_csv('data/smoking.csv')
            df = df.drop('ID', axis=1)

            le = LabelEncoder()
            categorical_cols = ['gender', 'oral', 'tartar']
            for col in categorical_cols:
                df[col] = le.fit_transform(df[col])

            self.X = df.drop('smoking', axis=1).values
            self.y = df['smoking'].values
            self.X_scaled = self.scaler.fit_transform(self.X)

            self.data_loaded = True
            self.data_label.config(
                text=f"Loaded: {len(self.X):,} samples, {self.X.shape[1]} features",
                foreground='green'
            )

            class_counts = np.bincount(self.y)
            info = "Class Distribution:\n"
            info += f"  Non-Smoker (0): {class_counts[0]:,}\n"
            info += f"  Smoker (1): {class_counts[1]:,}\n"
            info += f"\nTotal: {len(self.y):,} samples"

            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, info)

            self.update_topology_label()

            messagebox.showinfo("Success", "Dataset loaded successfully!")

        except Exception as e:
            messagebox.showerror(
                "Error", f"Error loading data: {str(e)}"
            )

    def create_model(self, random_state=42):
        layer_sizes = [
            self.X.shape[1],
            self.hidden1.get(),
            self.hidden2.get(),
            1
        ]
        return ArtificialNeuralNetwork(
            layer_sizes=layer_sizes,
            learning_rate=self.learning_rate.get(),
            iterations=self.iterations.get(),
            random_state=random_state
        )

    def k_fold_cross_validation(self, X, y, k=5, random_state=42):
        np.random.seed(random_state)
        n_samples = len(y)
        indices = np.arange(n_samples)
        np.random.shuffle(indices)

        fold_size = n_samples // k
        scores = []
        all_predictions = np.zeros(n_samples)

        for fold in range(k):
            test_start = fold * fold_size
            test_end = test_start + fold_size if fold < k - 1 else n_samples
            test_indices = indices[test_start:test_end]
            train_indices = np.concatenate([
                indices[:test_start], indices[test_end:]
            ])

            X_train, X_test = X[train_indices], X[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            model = self.create_model(random_state=random_state + fold)
            model.fit(X_train, y_train, verbose=False)
            y_pred = model.predict(X_test)

            acc = np.mean(y_pred == y_test)
            scores.append(acc)
            all_predictions[test_indices] = y_pred

        return np.array(scores), all_predictions.astype(int)

    def update_topology_label(self):
        if self.X is not None:
            topo = (
                f"Topology: {self.X.shape[1]} -> {self.hidden1.get()} "
                f"-> {self.hidden2.get()} -> 1"
            )
            self.topo_label.config(text=topo)

    def run_scenario(self):
        if not self.data_loaded:
            messagebox.showwarning(
                "Warning", "Please load the dataset first!"
            )
            return

        self.update_topology_label()
        scenario = self.scenario_var.get()

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Training model...\n")
        self.root.update()

        try:
            if scenario == 1:
                self.run_scenario1()
            elif scenario == 2:
                self.run_scenario2()
            elif scenario == 3:
                self.run_scenario3()
            elif scenario == 4:
                self.run_scenario4()
        except Exception as e:
            messagebox.showerror(
                "Error", f"Error during training: {str(e)}"
            )

    def run_scenario1(self):
        model = self.create_model()
        model.fit(self.X_scaled, self.y)
        y_pred = model.predict(self.X_scaled)
        acc = np.mean(y_pred == self.y)

        result = "SCENARIO 1: Training = Test\n"
        result += "=" * 35 + "\n\n"
        result += (
            f"Network Topology: {self.X.shape[1]}->{self.hidden1.get()}"
            f"->{self.hidden2.get()}->1\n"
        )
        result += "Activation: Sigmoid\n"
        result += "Loss: MSE\n\n"
        result += f"* Accuracy: {acc*100:.2f}%\n"

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, result)

        self.plot_confusion_matrix(self.y, y_pred, "Scenario 1")

    def run_scenario2(self):
        scores, y_pred = self.k_fold_cross_validation(
            self.X_scaled, self.y, k=5
        )

        result = "SCENARIO 2: 5-Fold Cross Validation\n"
        result += "=" * 35 + "\n\n"
        result += (
            f"Network Topology: {self.X.shape[1]}->{self.hidden1.get()}"
            f"->{self.hidden2.get()}->1\n\n"
        )
        result += "Fold Results:\n"
        for i, score in enumerate(scores, 1):
            result += f"  Fold {i}: {score*100:.2f}%\n"
        result += f"\n* Mean: {scores.mean()*100:.2f}%\n"
        result += f"  Std: +/-{scores.std()*100:.2f}%\n"

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, result)

        self.plot_confusion_matrix(self.y, y_pred, "Scenario 2 (5-Fold)")

    def run_scenario3(self):
        scores, y_pred = self.k_fold_cross_validation(
            self.X_scaled, self.y, k=10
        )

        result = "SCENARIO 3: 10-Fold Cross Validation\n"
        result += "=" * 35 + "\n\n"
        result += (
            f"Network Topology: {self.X.shape[1]}->{self.hidden1.get()}"
            f"->{self.hidden2.get()}->1\n\n"
        )
        result += "Fold Results:\n"
        for i, score in enumerate(scores, 1):
            result += f"  Fold {i:2d}: {score*100:.2f}%\n"
        result += f"\n* Mean: {scores.mean()*100:.2f}%\n"
        result += f"  Std: +/-{scores.std()*100:.2f}%\n"

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, result)

        self.plot_confusion_matrix(self.y, y_pred, "Scenario 3 (10-Fold)")

    def run_scenario4(self):
        seeds = [42, 123, 456, 789, 999]
        results = []

        result = "SCENARIO 4: 75-25 Split (5 Seeds)\n"
        result += "=" * 35 + "\n\n"
        result += (
            f"Network Topology: {self.X.shape[1]}->{self.hidden1.get()}"
            f"->{self.hidden2.get()}->1\n\n"
        )

        best_acc = 0
        best_data = None

        for seed in seeds:
            X_train, X_test, y_train, y_test = train_test_split(
                self.X_scaled, self.y,
                test_size=0.25,
                random_state=seed,
                stratify=self.y
            )
            model = self.create_model(random_state=seed)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = np.mean(y_pred == y_test)
            results.append(acc)
            result += f"  Seed {seed}: {acc*100:.2f}%\n"

            if acc > best_acc:
                best_acc = acc
                best_data = (y_test, y_pred)

        avg = np.mean(results)
        std = np.std(results)
        result += f"\n* Mean: {avg*100:.2f}%\n"
        result += f"  Std: +/-{std*100:.2f}%\n"

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, result)

        self.plot_confusion_matrix(
            best_data[0], best_data[1], "Scenario 4 (Best)"
        )

    def plot_confusion_matrix(self, y_true, y_pred, title):
        for widget in self.graph_frame.winfo_children():
            widget.destroy()

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(4, 3.5))
        im = ax.imshow(cm, cmap='Blues')

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Non-Smoker', 'Smoker'])
        ax.set_yticklabels(['Non-Smoker', 'Smoker'])

        for i in range(2):
            for j in range(2):
                text = ax.text(
                    j, i, f'{cm[i, j]:,}',
                    ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black',
                    fontsize=11
                )

        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix\n{title}')
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)

    def show_network(self):
        if not self.data_loaded:
            messagebox.showwarning(
                "Warning", "Please load the dataset first!"
            )
            return

        net_window = tk.Toplevel(self.root)
        net_window.title("Artificial Neural Network Architecture")
        net_window.geometry("700x500")

        fig, ax = plt.subplots(figsize=(8, 5))

        layers = [
            self.X.shape[1],
            self.hidden1.get(),
            self.hidden2.get(),
            1
        ]
        layer_names = [
            f'Input\n({layers[0]})',
            f'Hidden 1\n({layers[1]})',
            f'Hidden 2\n({layers[2]})',
            f'Output\n({layers[3]})'
        ]
        colors = ['#3498db', '#2ecc71', '#2ecc71', '#e74c3c']

        x_positions = [0.15, 0.4, 0.65, 0.9]
        max_display = [8, 6, 6, 1]

        node_positions = []

        for i, (layer_size, max_d, x, color, name) in enumerate(
            zip(layers, max_display, x_positions, colors, layer_names)
        ):
            positions = []
            n_display = min(layer_size, max_d)
            y_start = 0.5 + (n_display - 1) * 0.06

            for j in range(n_display):
                y = y_start - j * 0.12
                circle = plt.Circle(
                    (x, y), 0.025, color=color, ec='black', lw=2, zorder=10
                )
                ax.add_patch(circle)
                positions.append((x, y))

            if layer_size > max_d:
                ax.text(
                    x, y_start - max_d * 0.12, '...',
                    fontsize=16, ha='center', va='center'
                )

            ax.text(
                x, 0.08, name,
                fontsize=10, ha='center', va='center', fontweight='bold'
            )
            node_positions.append(positions)

        for i in range(len(node_positions) - 1):
            for pos1 in node_positions[i]:
                for pos2 in node_positions[i + 1]:
                    ax.plot(
                        [pos1[0], pos2[0]], [pos1[1], pos2[1]],
                        'gray', alpha=0.3, lw=0.5
                    )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(
            'Artificial Neural Network Architecture\n'
            'Activation: Sigmoid | Loss: MSE',
            fontsize=12, fontweight='bold'
        )

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, net_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)

    def compare_all(self):
        if not self.data_loaded:
            messagebox.showwarning(
                "Warning", "Please load the dataset first!"
            )
            return

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, "Running all scenarios...\n")
        self.root.update()

        results = {}

        # Scenario 1
        model1 = self.create_model()
        model1.fit(self.X_scaled, self.y)
        y_pred1 = model1.predict(self.X_scaled)
        results['Training=Test'] = np.mean(y_pred1 == self.y)

        # Scenario 2
        scores2, _ = self.k_fold_cross_validation(
            self.X_scaled, self.y, k=5
        )
        results['5-Fold CV'] = scores2.mean()

        # Scenario 3
        scores3, _ = self.k_fold_cross_validation(
            self.X_scaled, self.y, k=10
        )
        results['10-Fold CV'] = scores3.mean()

        # Scenario 4
        seeds = [42, 123, 456, 789, 999]
        accs = []
        for seed in seeds:
            X_train, X_test, y_train, y_test = train_test_split(
                self.X_scaled, self.y,
                test_size=0.25,
                random_state=seed,
                stratify=self.y
            )
            model4 = self.create_model(random_state=seed)
            model4.fit(X_train, y_train)
            accs.append(np.mean(model4.predict(X_test) == y_test))
        results['75-25'] = np.mean(accs)

        # Display results
        result = "ALL SCENARIOS COMPARISON\n"
        result += "=" * 35 + "\n\n"
        for name, acc in results.items():
            result += f"{name:<15}: {acc*100:.2f}%\n"

        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, result)

        # Plot
        for widget in self.graph_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(4, 3.5))
        colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c']
        bars = ax.bar(
            results.keys(),
            [v*100 for v in results.values()],
            color=colors,
            edgecolor='black'
        )
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Scenario Comparison')
        min_acc = min(results.values()) * 100
        max_acc = max(results.values()) * 100
        ax.set_ylim(max(0, min_acc - 5), min(100, max_acc + 5))
        ax.grid(axis='y', alpha=0.3)

        for bar, acc in zip(bars, results.values()):
            ax.text(
                bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.3,
                f'{acc*100:.1f}%',
                ha='center', fontsize=9, fontweight='bold'
            )

        plt.xticks(rotation=15, ha='right')
        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, self.graph_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plt.close(fig)


if __name__ == "__main__":
    root = tk.Tk()
    app = SmokingClassifierGUI(root)
    root.mainloop()
