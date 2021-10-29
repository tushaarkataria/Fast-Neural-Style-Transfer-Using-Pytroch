import torch
import torch.nn as nn
import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2
from 



def trainingLoop(loader, styleImage,model,optimizer,nepochs,alpha,alphatv,directoryName,webpage):
    train_transform = A.Compose([
                                 A.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
                                 ToTensorV2(),
                                ])
    ## VGG model to GPU
    vggmodel = VGGModule()
    vggmodel = vggmodel.to(device=cpu,dtype=dtype)
    vggmodel.eval()

    # Style Image Precalculations
    styleImage = torch.tensor(styleImage)
    styleImage = styleImage.to(device=cpu,dtype=dtype)
    mean = torch.tensor([0.485,0.456,0.406]).to(device=cpu,dtype=dtype)
    std  = torch.tensor([0.229,0.224,0.225]).to(device=cpu,dtype=dtype)
    styleImage   = styleImage-mean/std
    styleImage = styleImage.permute(2,0,1)
    styleImage = styleImage.unsqueeze(0)
    styleImage   = styleImage.to(device=cpu,dtype=dtype)
    ## Style Image Activations and Gram Matrix
    relu1_2_s, relu2_2_s, relu3_3_s, relu4_3_s = vggmodel(styleImage)
    G_relu1_2_s = gram_matrix(relu1_2_s)
    G_relu2_2_s = gram_matrix(relu2_2_s)
    G_relu3_3_s = gram_matrix(relu3_3_s)
    G_relu4_3_s = gram_matrix(relu4_3_s)

    ## MSE loss and weights for different losses
    contentLoss = nn.MSELoss()
    k=0
    fullLossArray=[]
    contentLossArray =[]
    styleLossArray =[]
    tvLossArray =[]
    for e in range(nepochs):
        for temp in loader:
            temp = temp.to(device=cpu,dtype=dtype)
            G_relu1_2_s = G_relu1_2_s.to(device=cpu,dtype=dtype)
            G_relu2_2_s = G_relu2_2_s.to(device=cpu,dtype=dtype)
            G_relu3_3_s = G_relu3_3_s.to(device=cpu,dtype=dtype)
            G_relu4_3_s = G_relu4_3_s.to(device=cpu,dtype=dtype)
            optimizer.zero_grad(set_to_none=True)
            model = model.to(device=cpu)    
            model.train()  
            output  = model(temp)
            relu1_2_o, relu2_2_o, relu3_3_o, relu4_3_o = vggmodel(output)
            relu1_2_c, relu2_2_c, relu3_3_c, relu4_3_c = vggmodel(temp)
            G_relu1_2_o = gram_matrix1(relu1_2_o)
            G_relu2_2_o = gram_matrix1(relu2_2_o)
            G_relu3_3_o = gram_matrix1(relu3_3_o)
            G_relu4_3_o = gram_matrix1(relu4_3_o)
            G_relu1_2_o = G_relu1_2_o.to(device=cpu,dtype=dtype)
            G_relu2_2_o = G_relu2_2_o.to(device=cpu,dtype=dtype)
            G_relu3_3_o = G_relu3_3_o.to(device=cpu,dtype=dtype)
            G_relu4_3_o = G_relu4_3_o.to(device=cpu,dtype=dtype)
            style_loss = torch.norm(G_relu1_2_s-G_relu1_2_o,p='fro')**2 + torch.norm(G_relu2_2_s-G_relu2_2_o,p='fro')**2 + torch.norm(G_relu3_3_s-G_relu3_3_o,p='fro')**2 + torch.norm(G_relu4_3_o-G_relu4_3_s,p='fro')**2
            content_loss = contentLoss(relu3_3_o, relu3_3_c)
            tvloss       =  total_variation_loss(output)
            loss   = content_loss + alpha*style_loss +  alphatv * tvloss 
            loss.backward(retain_graph=True)
            optimizer.step()
            if(k%100==0 and k>0):
                contentLossArray.append(content_loss.item())
                styleLossArray.append(style_loss.item())
                fullLossArray.append(loss.item())
                tvLossArray.append(tvloss.item())
                best_model = copy.deepcopy(model.state_dict())
                torch.save(best_model,directoryName+'/best_model_batch'+str(k)+'.pt')
                print("Full loss: ",loss.item())
                #image =output[0].clone().detach().squeeze()
                #image = image.permute(1,2,0)
                #image = image.cpu()
                #io.imsave(directoryName+"/GeneratedImage"+str(k)+'.png',image)
                contentImage = io.imread('chicago.jpg')  
                contentImage = contentImage/np.max(contentImage) 
                contentImage = train_transform(image=contentImage)
                contentImage = contentImage['image'] 
                contentImage = contentImage.unsqueeze(0) 
                contentImage = contentImage.to('cuda')
                output = model(contentImage)
                output = output.squeeze()
                output = output.permute(1,2,0)
                output = output.squeeze().detach().cpu().numpy()
                io.imsave(directoryName+'/images/Chicago_output'+str(k)+'.png',output)
                contentImage = io.imread('hoovertowernight.jpg')  
                contentImage = contentImage/np.max(contentImage) 
                contentImage = train_transform(image=contentImage)
                contentImage = contentImage['image'] 
                contentImage = contentImage.unsqueeze(0) 
                contentImage = contentImage.to('cuda')
                output = model(contentImage)
                output = output.squeeze()
                output = output.permute(1,2,0)
                output = output.squeeze().detach().cpu().numpy()
                io.imsave(directoryName+'/images/hoovertower_output'+str(k)+'.png',output)
                ## Writing to Webpage
                visuals = OrderedDict()
                visuals={'chicago'+str(k):'Chicago_output'+str(k)+'.png','hoover'+str(k):'hoovertower_output'+str(k)+'.png'}
                save_images(webpage, visuals, directoryName+'/images', aspect_ratio=1,width=512)
                webpage.save()
            k=k+1
    best_model = copy.deepcopy(model.state_dict())
    torch.save(best_model,directoryName+'/best_model_batch.pt')
    fig, ax = plt.subplots(2,2)
    fig.set_size_inches(28,20)
    ax[0][0].plot(fullLossArray)
    ax[0][0].set_title("Total Loss")
    ax[0][1].plot(styleLossArray)
    ax[0][1].set_title("Style Loss")
    ax[1][0].plot(contentLossArray)
    ax[1][0].set_title("content Loss")
    ax[1][1].plot(tvLossArray)
    ax[1][1].set_title("TV Loss")
    plt.savefig(directoryName+"/FullLoss.png")    
    return loss

