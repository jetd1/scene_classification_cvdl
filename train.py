# Skeleton code
# Xiaoshuai 'Jet' Zhang, jet@pku.edu.cn
# HW01 for CVDL course of PKU

def TrainNet(net, cri, opt, cp_dir, epoch_range, t_loader, v_loader, cuda=True):
    ensure_exists(cp_dir)
    fout = open(os.path.join(cp_dir,'train.log'),'a')

    for epoch in epoch_range:
        net.train()
        running_loss = 0
        running_acc = 0
        start_time = time.time()
        tl = len(t_loader)
        
        i = 0
        for data in t_loader:
            in_img = var(data[0])
            labels = var(data[1])
            
            if cuda:
                in_img, labels = in_img.cuda(), labels.cuda()

            opt.zero_grad()
            ret = net(in_img)
            loss = cri(ret, labels)
            loss.backward()
#           nn.utils.clip_grad_norm(net.parameters(), 5.0)
            opt.step()

            running_loss += loss.data[0]
        
            _, predict = torch.max(ret, 1)
            num_correct = (predict == labels).sum()
            running_acc += num_correct.data[0]
            
            i += 1

            print('[Running epoch %2d, batch %4d] loss: %.4f, acc: %.4f' %
                  (epoch + 1, i,
                   running_loss / i,
                   running_acc / (i * t_loader.batch_size)), end='\r')
        
        
        timestamp = time.time()
        print('[epoch % 2d, %d Batches] loss: %.4f, time: %5ds               ' %
              (epoch + 1, tl, running_loss / tl,
               timestamp - start_time), end='\n')
        print('[timestamp %d, epoch %2d] loss: %.4f, time: %6ds              ' %
              (timestamp, epoch + 1, running_loss / tl,
                timestamp - start_time), end='\n', file=fout)
        torch.save(net.state_dict(), os.path.join(cp_dir, str(epoch+1)+'-'+str(timestamp)))

        
        net.eval()
        val_loss = 0
        val_acc = 0
        val_acc3 = 0
        vl = len(v_loader)
        vdl = len(v_loader.dataset)
        j = 0
        for data in tqdm(v_loader, file=sys.stdout):
            in_img = var(data[0]).cuda()
            labels = var(data[1]).cuda()

            if cuda:
                in_img, labels = in_img.cuda(), labels.cuda()

            ret = net(in_img)
            
            loss = cri(ret, labels)
            val_loss += loss.data[0]
            
            _, predict = torch.max(ret, 1)
            num_correct = (predict == labels).sum()
            val_acc += num_correct.data[0]
            
            _, p3 = torch.topk(ret, 3)
            labels = labels.view(-1,1).expand_as(p3)
            val_acc3 += (labels == p3).sum().data[0]
            
            j += 1

        print('val_loss: %.4f, val_acc: %.4f, val_acc3: %.4f\n'
              % (val_loss / vl,
                 val_acc / vdl,
                 val_acc3 / vdl), end='\n')

        print('val_loss: %.4f, val_acc: %.4f, val_acc3: %.4f\n'
              % (val_loss / vl,
                 val_acc / vdl,
                 val_acc3 / vdl), end='\n', file=fout)

        fout.flush()
