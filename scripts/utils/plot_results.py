import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt


def read_log(log_path):
    val_loss = []
    val_acc = []

    f = open(log_path,'r')
    for i, line in enumerate (f):
        if "Final" in line:
            break
        idx_val_loss = line.find('val_loss')
        idx_val_acc1 = line.find('val_acc1')
        idx_val_acc5 = line.find('val_acc5')

        val_loss.append(float(line[idx_val_loss+11:idx_val_acc1-3]))
        val_acc.append(float(line[idx_val_acc1+11:idx_val_acc5-3]))
    return val_loss, val_acc

def read_nohup(nohup_path):
    val_loss = []
    val_acc = []

    f = open(nohup_path,'r')
    for i, line in enumerate (f):
        if "Acc@1" in line:
            idx_val_loss = line .find('loss')
            idx_val_acc1 = line.find('Acc@1')
            idx_val_acc5 = line.find('Acc@5')

            val_loss.append(float(line[idx_val_loss+4:]))
            val_acc.append(float(line[idx_val_acc1+6:idx_val_acc5-2]))
    return val_loss[:-1], val_acc[:-1]

scratch_add = "VideoMAE/results/finetune_Allclass/log.txt"
BB_add = "VideoMAE/results/finetune_Allclass_BB_VideoMAE_scratch(50)/log.txt"
scratch_plus_100_add = "VideoMAE/results/finetune_Allclass_VideoMAE(900)_50epoch/log.txt"
BB_plus_100_add = "nohup_finetune_BB_pretrained(100epoch)_Allclass_Init_VideoMAE_pretained.out"

scratch_val_loss, scratch_val_acc = read_log(scratch_add)
BB_val_loss, BB_val_acc = read_log(BB_add)
scratch_plus_100_val_loss, scratch_plus_100_val_acc = read_log(scratch_plus_100_add)
BB_plus_100_val_loss, BB_plus_100_val_acc = read_nohup(BB_plus_100_add)

# plot loss
plt.figure()
plt.plot(scratch_val_loss, label='scratch')
plt.plot(BB_val_loss, label='BB')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('val loss')
plt.savefig('val_loss.png', dpi=300)

plt.figure()
plt.plot(scratch_plus_100_val_loss, label='scratch+100')
plt.plot(BB_plus_100_val_loss, label='BB+100')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('val loss')
plt.savefig('val_loss_100.png', dpi=300)

# plot acc
plt.figure()
plt.plot(scratch_val_acc, label='scratch')
plt.plot(BB_val_acc, label='BB')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('val acc')
plt.savefig('val_acc.png', dpi=300)

plt.figure()
plt.plot(scratch_plus_100_val_acc, label='scratch+100')
plt.plot(BB_plus_100_val_acc, label='BB+100')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('val acc')
plt.savefig('val_acc_100.png', dpi=300)

# plot loss first 5 epochs
plt.figure()
plt.plot(scratch_val_loss[:5], label='scratch')
plt.plot(BB_val_loss[:5], label='BB')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('val loss')
plt.savefig('val_loss_5.png', dpi=300)

plt.figure()
plt.plot(scratch_plus_100_val_loss[:5], label='scratch+100')
plt.plot(BB_plus_100_val_loss[:5], label='BB+100')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('val loss')
plt.savefig('val_loss_100_5.png', dpi=300)

# plot acc first 5 epochs
plt.figure()
plt.plot(scratch_val_acc[:5], label='scratch')
plt.plot(BB_val_acc[:5], label='BB')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('val acc')
plt.savefig('val_acc_5.png', dpi=300)

plt.figure()
plt.plot(scratch_plus_100_val_acc[:5], label='scratch+100')
plt.plot(BB_plus_100_val_acc[:5], label='BB+100')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('val acc')
plt.savefig('val_acc_100_5.png', dpi=300)




