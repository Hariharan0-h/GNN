"""
Research-grade GraphSAGE Trainer
"""

import os
import copy

import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch_geometric.loader import DataLoader

from sklearn.model_selection import train_test_split

from gnn.model import GraphSAGEModel


class GraphTrainer:

    def __init__(self):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = GraphSAGEModel().to(
            self.device
        )

        self.val_loader = None
        self.test_loader = None

        # Training history for paper experiments
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": []
        }


    def train(
        self,
        dataset,
        epochs=100,
        batch_size=64,
        lr=1e-3
    ):

        train_set, temp_set = train_test_split(
            dataset,
            test_size=0.20,
            random_state=42
        )

        val_set, test_set = train_test_split(
            temp_set,
            test_size=0.50,
            random_state=42
        )

        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True
        )

        self.val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False
        )

        self.test_loader = DataLoader(
            test_set,
            batch_size=batch_size,
            shuffle=False
        )

        optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=1e-4
        )

        labels = []

        for data in train_set:
            labels.append(data.y)

        labels = torch.cat(labels)

        class_counts = torch.bincount(labels)

        class_weights = (
            1.0 /
            class_counts.float()
        )

        class_weights = (
            class_weights /
            class_weights.sum()
        )

        print(
            "Class distribution:",
            class_counts.tolist()
        )

        print(
            "Class weights:",
            class_weights.tolist()
        )

        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(
                self.device
            )
        )

        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5
        )

        os.makedirs(
            "gnn/checkpoints",
            exist_ok=True
        )

        checkpoint_path = (
            "gnn/checkpoints/"
            "best_graphsage_model.pth"
        )

        best_val_loss = float("inf")

        best_model = copy.deepcopy(
            self.model.state_dict()
        )

        patience = 15
        wait = 0

        for epoch in range(epochs):

            # -------------------------
            # TRAIN
            # -------------------------

            self.model.train()

            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for data in train_loader:

                data = data.to(
                    self.device
                )

                optimizer.zero_grad()

                output = self.model(
                    data.x,
                    data.edge_index
                )

                loss = criterion(
                    output,
                    data.y
                )

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0
                )

                optimizer.step()

                train_loss += loss.item()

                predictions = torch.argmax(
                    output,
                    dim=1
                )

                train_correct += (
                    predictions == data.y
                ).sum().item()

                train_total += data.y.size(0)

            train_accuracy = (
                train_correct /
                train_total
            )

            # -------------------------
            # VALIDATION
            # -------------------------

            self.model.eval()

            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():

                for data in self.val_loader:

                    data = data.to(
                        self.device
                    )

                    output = self.model(
                        data.x,
                        data.edge_index
                    )

                    loss = criterion(
                        output,
                        data.y
                    )

                    val_loss += loss.item()

                    predictions = torch.argmax(
                        output,
                        dim=1
                    )

                    val_correct += (
                        predictions == data.y
                    ).sum().item()

                    val_total += data.y.size(0)

            val_accuracy = (
                val_correct /
                val_total
            )

            scheduler.step(
                val_loss
            )

            # Save history
            self.history[
                "train_loss"
            ].append(train_loss)

            self.history[
                "val_loss"
            ].append(val_loss)

            self.history[
                "train_accuracy"
            ].append(train_accuracy)

            self.history[
                "val_accuracy"
            ].append(val_accuracy)

            print(
                f"Epoch {epoch+1:03d} | "
                f"Train Loss={train_loss:.4f} | "
                f"Val Loss={val_loss:.4f} | "
                f"Train Acc={train_accuracy:.4f} | "
                f"Val Acc={val_accuracy:.4f}"
            )

            # -------------------------
            # BEST MODEL
            # -------------------------

            if val_loss < best_val_loss:

                best_val_loss = val_loss

                wait = 0

                best_model = copy.deepcopy(
                    self.model.state_dict()
                )

                torch.save(
                    best_model,
                    checkpoint_path
                )

                print(
                    "Best model saved"
                )

            else:

                wait += 1

                if wait >= patience:

                    print(
                        "\nEarly stopping triggered\n"
                    )

                    break

        # Restore best model
        self.model.load_state_dict(
            best_model
        )

        print(
            "\nTraining completed"
        )

        print(
            f"Best validation loss: "
            f"{best_val_loss:.4f}"
        )

        return self.model