import torch 
import torch.nn as nn

class trainner:
    def Train(model,device, criterion, data_loader, val_loader , optimizer, num_epochs ,scheduler=None ,scheduler2=None , model_path='model.ckpt'):
        """Simple training loop for a PyTorch model.""" 
        best_val_acc=0
        # Make sure model is in training mode.
        model.train()
        x=torch.tensor([2,2])
        optimizer.zero_grad()

        # Move model to the device (CPU or GPU).
        model.to(device)
        
        # Exponential moving average of the loss.
        ema_loss = None
        
        print('----- Training -----')
        # Loop over epochs.
        for epoch in range(num_epochs):
            total=0
            train_running_correct=0
              # Loop over data.
            for batch_idx, (features, target) in enumerate(data_loader):
                # Forward pass.
                output = model(features.to(device))
                target=target.to(device)
                loss = criterion(output.to(device), target)


                  # Backward pass.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                _, preds = torch.max(output.data, 1)
                total += target.size(0)
                train_running_correct += (preds == target).sum().item()

              # NOTE: It is important to call .item() on the loss before summing.
                if ema_loss is None:
                    ema_loss = loss.item()
                else:
                    ema_loss += (loss.item() - ema_loss) * 0.01 

            
            
            train_acc = 100. * train_running_correct / total
            print('Epoch: {} \tLoss: {:.6f}  , train acc = {}%  {}/{}'.format(epoch+1, ema_loss  ,train_acc,train_running_correct , total ),)

              # Print out progress the end of epoch.
            if scheduler2 and epoch ==10:
                optimizer = torch.optim.RMSprop(model.parameters(), lr=0.00001 , momentum=0.9)

            if scheduler2 and epoch>=10:
                scheduler2.step()
            elif scheduler:
                scheduler.step()

            # Print out progress the end of epoch.
            val_acc = model.Test( val_loader)


            if val_acc>best_val_acc and val_acc>88.0:
                print(f'validation accuracy increased from {best_val_acc} to {val_acc}  , saving the model....')
                if model_path : torch.save(model.state_dict() , model_path)
                best_val_acc=val_acc
            print('##'*20)
                
                
    # def get_lr(optimizer):
    #     for param_group in optimizer.param_groups:
    #         return param_group['lr']


    def Test(model, data_loader , txt='validation'):
        """Measures the accuracy of a model on a data set.""" 
        # Make sure the model is in evaluation mode.
        model.eval()
        correct = 0
        total=0
        print('----- Model Evaluation -----')
        # We do not need to maintain intermediate activations while testing.
        with torch.no_grad():
            
            # Loop over test data.
            for features, target in data_loader:
                total+=target.shape[0]
                # Forward pass.
                output = model(features.to(device))
                
                # Get the label corresponding to the highest predicted probability.
                pred = output.argmax(dim=1, keepdim=True)
                
                # Count number of correct predictions.
                correct += pred.cpu().eq(target.view_as(pred)).sum().item()
        model.train()
        # Print test accuracy.
        percent = 100. * correct / len(data_loader.sampler)
        print(f'{txt} accuracy: {correct} / {len(data_loader.sampler)} ({percent:.2f}%)')
        return percent

    def print_layers(model):
      for layer in model.children():
        print(layer)