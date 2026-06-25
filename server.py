import torch
from .client import SimpleMLP

class FLServer:
    """
    Central Server:
    Manages global model state, maintains global versioning, and handles model broadcasting.
    """
    def __init__(self, device='cpu'):
        self.device = device
        # Initialize a global MLP model
        self.global_model = SimpleMLP().to(self.device)
        self.global_version = 0
        
    def get_global_model_state(self):
        """
        Retrieves the current global model weight dictionary for broadcasting to clients.
        """
        return self.global_model.state_dict()
        
    def update_global_model(self, new_state_dict):
        """
        Updates the global model with aggregated parameters and increments the version ID.
        """
        self.global_model.load_state_dict(new_state_dict)
        self.global_version += 1

    def evaluate(self, test_loader, criterion=torch.nn.CrossEntropyLoss()):
        """
        Evaluates the current accuracy of the global model on the test set.
        :param test_loader: DataLoader for the test set
        :param criterion: Loss function
        :return: (accuracy, avg_loss)
        """
        self.global_model.eval()
        correct = 0
        total = 0
        test_loss = 0.0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.global_model(images)
                
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        accuracy = correct / total
        avg_loss = test_loss / len(test_loader)
        return accuracy, avg_loss
