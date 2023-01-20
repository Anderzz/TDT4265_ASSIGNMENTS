import numpy as np
import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images
from trainer import BaseTrainer
from task3a import cross_entropy_loss, SoftmaxModel, one_hot_encode
np.random.seed(0)


def calculate_accuracy(X: np.ndarray, targets: np.ndarray, model: SoftmaxModel) -> float:
    """
    Args:
        X: images of shape [batch size, 785]
        targets: labels/targets of each image of shape: [batch size, 10]
        model: model of class SoftmaxModel
    Returns:
        Accuracy (float)
    """
    # TODO: Implement this function (task 3c)
    accuracy = 0
    total = X.shape[0]
    res = model.forward(X)
    res = np.argmax(res, axis=1)
    targets = np.argmax(targets, axis=1)
    num_correct = (targets.squeeze() == res.squeeze()).sum()
    accuracy = num_correct / total
    return accuracy


class SoftmaxTrainer(BaseTrainer):

    def train_step(self, X_batch: np.ndarray, Y_batch: np.ndarray):
        """
        Perform forward, backward and gradient descent step here.
        The function is called once for every batch (see trainer.py) to perform the train step.
        The function returns the mean loss value which is then automatically logged in our variable self.train_history.

        Args:
            X: one batch of images
            Y: one batch of labels
        Returns:
            loss value (float) on batch
        """
        # TODO: Implement this function (task 3b)
        loss = 0
        a = self.model.forward(X_batch)
        self.model.backward(X_batch, a, Y_batch)
        self.model.w = self.model.w - self.model.grad * self.learning_rate # gradient descent
        self.model.zero_grad() # reset gradient
        loss = cross_entropy_loss(Y_batch, a)
        return loss

    def validation_step(self):
        """
        Perform a validation step to evaluate the model at the current step for the validation set.
        Also calculates the current accuracy of the model on the train set.
        Returns:
            loss (float): cross entropy loss over the whole dataset
            accuracy_ (float): accuracy over the whole dataset
        Returns:
            loss value (float) on batch
            accuracy_train (float): Accuracy on train dataset
            accuracy_val (float): Accuracy on the validation dataset
        """
        # NO NEED TO CHANGE THIS FUNCTION
        logits = self.model.forward(self.X_val)
        loss = cross_entropy_loss(self.Y_val, logits)

        accuracy_train = calculate_accuracy(
            self.X_train, self.Y_train, self.model)
        accuracy_val = calculate_accuracy(
            self.X_val, self.Y_val, self.model)
        return loss, accuracy_train, accuracy_val


def main():
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = 0.01
    batch_size = 128
    l2_reg_lambda = 0
    shuffle_dataset = True
    weights = []
    l2_norms = []

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    # ANY PARTS OF THE CODE BELOW THIS CAN BE CHANGED.

    # Intialize model
    model = SoftmaxModel(l2_reg_lambda)
    # Train model
    trainer = SoftmaxTrainer(
        model, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)

    print("Final Train Cross Entropy Loss:",
          cross_entropy_loss(Y_train, model.forward(X_train)))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model.forward(X_val)))
    print("Final Train accuracy:", calculate_accuracy(X_train, Y_train, model))
    print("Final Validation accuracy:", calculate_accuracy(X_val, Y_val, model))

    plt.ylim([0.2, .8])
    utils.plot_loss(train_history["loss"],
                    "Training Loss", npoints_to_average=10)
    utils.plot_loss(val_history["loss"], "Validation Loss")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.savefig("task3b_softmax_train_loss.png")
    plt.show()

    # Plot accuracy
    plt.ylim([0.89, .93])
    utils.plot_loss(train_history["accuracy"], "Training Accuracy")
    utils.plot_loss(val_history["accuracy"], "Validation Accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("task3b_softmax_train_accuracy.png")
    plt.show()

    # Train a model with L2 regularization (task 4b)

    model1 = SoftmaxModel(l2_reg_lambda=2.0)
    trainer = SoftmaxTrainer(
        model1, learning_rate, batch_size, shuffle_dataset,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_reg01, val_history_reg01 = trainer.train(num_epochs)

    print("Final Train Cross Entropy Loss lambda = 2:",
          cross_entropy_loss(Y_train, model1.forward(X_train)))
    print("Final Validation Cross Entropy Loss lambda = 2:",
          cross_entropy_loss(Y_val, model1.forward(X_val)))
    print("Final Train accuracy lambda = 2:", calculate_accuracy(X_train, Y_train, model1))
    print("Final Validation accuracy lambda = 2:", calculate_accuracy(X_val, Y_val, model1))

    plt.ylim([1.2, 2.2])
    utils.plot_loss(train_history_reg01["loss"],
                    "Training Loss lambda=2", npoints_to_average=10)
    utils.plot_loss(val_history_reg01["loss"], "Validation Loss lambda=2")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    plt.savefig("task4b_softmax_train_loss_lambda2.png")
    plt.show()
    # You can finish the rest of task 4 below this point.

    # 4b)

    # get the weights of the models
    weights.append(model.w)
    weights.append(model1.w)
    weights = np.array(weights)

    # plot them
    f, axarr = plt.subplots(2,10, figsize=(15,3))
    f.subplots_adjust(wspace=0, hspace=0)
    for i in range(10):
        axarr[0,i].imshow(weights[0][:-1,i].reshape(28,28), cmap='gray')
        axarr[0,i].axis('off')
        axarr[1,i].imshow(weights[1][:-1,i].reshape(28,28), cmap='gray')
        axarr[1,i].axis('off') 

    plt.savefig("task4b_softmax_weight_with_without_reg.png")

    # 4c)

    l2_lambdas = [2, .2, .02, .002]
    plt.figure()
    for lam in l2_lambdas:
        model = SoftmaxModel(lam)
        trainer = SoftmaxTrainer(
            model, learning_rate, batch_size, shuffle_dataset,
            X_train, Y_train, X_val, Y_val,
        )
        train_history, val_history = trainer.train(num_epochs)
        l2_norms.append(np.linalg.norm(model.w, 2))
        utils.plot_loss(val_history["accuracy"], f"lambda={lam}")
    plt.ylim([0.75, .93])
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.savefig("task4c_diff_lambdas.png")
    plt.show()
    plt.clf()

    # 4e
    print(f"norms: \n{l2_norms}\n\n lambdas \n{l2_lambdas}")
    plt.figure()
    plt.plot(l2_lambdas, l2_norms)
    plt.xlabel("lambda")
    plt.ylabel("L2 norm")
    plt.savefig("task4e_l2reg_norms.png")
    plt.show()

if __name__ == "__main__":
    main()
