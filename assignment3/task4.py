import torch
import pathlib
import matplotlib.pyplot as plt
import utils
import torchvision
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer, compute_loss_and_accuracy
from torchsummary import summary

class Model(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        self.num_classes = num_classes
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512, num_classes)
        
        for param in self.model.parameters(): # freeze
            param.requires_grad = False

        for param in self.model.fc.parameters(): # unfreeze fc layer
            param.requires_grad = True
        
        for param in self.model.layer4.parameters(): # unfreeze last 5 layers
            param.requires_grad = True
        

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        # # TODO: Implement this function (Task  2a)
        # batch_size = x.shape[0]
        # x = self.feature_extractor(x)#sol
        # x = x.view(-1, self.num_output_features)#sol
        # out = x
        # out = self.classifier(x)#sol
        # expected_shape = (batch_size, self.num_classes)
        # assert out.shape == (batch_size, self.num_classes),\
        #     f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        # return out
        x = self.model(x)
        return x


def create_plots(trainer: Trainer, name: str):
    plot_path = pathlib.Path("plots")
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}_plot.png"))
    plt.show()


def main():
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result! 
    utils.set_seed(0)
    epochs = 10
    batch_size = 32
    #learning_rate = 1e-3 # for adam
    learning_rate = 5e-4
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size, resnet=True)
    model = Model(image_channels=3, num_classes=10)
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )

    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=learning_rate, weight_decay=1e-4)
    trainer.train()
    trainer.load_best_model()
    datasets =  [trainer.dataloader_train, trainer.dataloader_val, trainer.dataloader_test]
    losses = []
    accuracies = []
    for dataloader in datasets:
        avg_loss, accuracy = compute_loss_and_accuracy(dataloader, trainer.model, trainer.loss_criterion)
        losses.append(avg_loss)
        accuracies.append(accuracy)
    print(f"Train loss: {losses[0]}, Train accuracy: {accuracies[0]}")
    print(f"Validation loss: {losses[1]}, Validation accuracy: {accuracies[1]}")
    print(f"Test loss: {losses[2]}, Test accuracy: {accuracies[2]}")

    summary(model, (3, 224, 224))
    create_plots(trainer, "task4")


if __name__ == "__main__":
    main()