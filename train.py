import json
import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
from torch import utils
from lifelines.utils.concordance import concordance_index
from pycox.models.loss import cox_ph_loss
from torch.utils.data import DataLoader
from datetime import datetime
import torchvision.transforms as transforms
from monai.transforms import RandFlip, NormalizeIntensity, Resize
from model import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--base', type=str, help='working base directory', default="YOUR/WORKING/PATH/")
parser.add_argument('--label', type=str, help='a .csv file recorded time and event,'
                                              ' the first column should be sample name', default="os_info.csv")
parser.add_argument('--splits', type=str, help='a .json file to split the data into training, '
                                               'validation, and test set', default="dataset_split.json")
parser.add_argument('--model', type=str, help='Rad-D or Rad-S', default="Rad-D")
parser.add_argument('--max_epoch', type=int, help='max epoch', default=200)
parser.add_argument('--lr', type=float, help='learning rate', default=5e-5)

args = parser.parse_args()


class SurvivalData(utils.data.Dataset):
    def __init__(self, mode="train", transform=None):
        super(SurvivalData, self).__init__()
        self.mode = mode
        with open(os.path.join(args.base, args.splits), "r") as ff:
            self.patients = json.load(ff)[mode]
        self.liver_path = os.path.join(args.base, "liver_img")
        self.lung_path = os.path.join(args.base, "lung_img")
        self.label = pd.read_csv(os.path.join(args.base, args.label), index_col=0)
        self.transform = transform

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, item):
        patient = self.patients[item]

        liver_files = list(filter(lambda x: x.startswith(patient), os.listdir(self.liver_path)))
        lung_files = list(filter(lambda x: x.startswith(patient), os.listdir(self.lung_path)))
        baseline = min(liver_files, key=lambda visit: datetime.strptime(visit[11:19], "%Y%m%d"))[:19]
        liver_choose = list(filter(lambda x: x.startswith(baseline), liver_files))
        lung_choose = list(filter(lambda x: x.startswith(baseline), lung_files))

        liver_imgs, lung_imgs = [], []

        for file in liver_choose:
            img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.liver_path, file)))
            if self.transform:
                img = self.transform(img)
            liver_imgs.append(img[0])
        liver_imgs = torch.stack(liver_imgs)  # 3*1*512*512

        if len(lung_choose) == 3:
            for file in lung_choose:
                img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.lung_path, file)))
                if self.transform:
                    img = self.transform(img)
                lung_imgs.append(img[0])
            lung_imgs = torch.stack(lung_imgs)
        if len(lung_choose) < 3:
            append_num = int(3 - len(lung_choose))
            for file in lung_choose + ["normal_lung.nii.gz",
                                       "normal_lung.nii.gz",
                                       "normal_lung.nii.gz"][:append_num]:
                img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.lung_path, file)))
                if self.transform:
                    img = self.transform(img)
                lung_imgs.append(img[0])
            lung_imgs = torch.stack(lung_imgs)

        select_fp = set([x[:19] for x in liver_files])
        followup_1_files = select_fp - {baseline}
        if len(followup_1_files) != 0:
            followup_1 = min(followup_1_files, key=lambda visit: datetime.strptime(visit[11:19], "%Y%m%d"))[:19]
            liver_choose_1 = list(filter(lambda x: x.startswith(followup_1), liver_files))
            lung_choose_1 = list(filter(lambda x: x.startswith(followup_1), lung_files))
            liver_imgs_followup, lung_imgs_followup = [], []

            for file in liver_choose_1:
                img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.liver_path, file)))
                if self.transform:
                    img = self.transform(img)
                liver_imgs_followup.append(img[0])
            liver_imgs_followup = torch.stack(liver_imgs_followup)  # 3*1*512*512

            if len(lung_choose_1) == 3:
                for file in lung_choose_1:
                    img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.lung_path, file)))
                    if self.transform:
                        img = self.transform(img)
                    lung_imgs_followup.append(img[0])
                lung_imgs_followup = torch.stack(lung_imgs_followup)
            if len(lung_choose_1) < 3:
                append_num = int(3 - len(lung_choose_1))
                for file in lung_choose_1 + ["normal_lung.nii.gz",
                                             "normal_lung.nii.gz",
                                             "normal_lung.nii.gz"][:append_num]:
                    img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.lung_path, file)))
                    if self.transform:
                        img = self.transform(img)
                    lung_imgs_followup.append(img[0])
                lung_imgs_followup = torch.stack(lung_imgs_followup)
        else:
            liver_imgs_followup = liver_imgs
            lung_imgs_followup = lung_imgs

        time_label = torch.tensor(self.label.loc[patient, "time"])
        event_label = torch.tensor(self.label.loc[patient, "event"])

        return liver_imgs, lung_imgs, liver_imgs_followup, lung_imgs_followup, time_label, event_label


transform_train = transforms.Compose([
    Resize(spatial_size=(224, 224)),
    RandFlip(prob=0.5),
    NormalizeIntensity()
])

transform_val = transforms.Compose([
    Resize(spatial_size=(224, 224)),
    NormalizeIntensity()
])


save_ckpt_path = os.path.join(args.base, f"ckpt_{args.model}")
os.makedirs(save_ckpt_path, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.model == "Rad-D":
    model = PrognosisModelD().to(device)
elif args.model == "Rad-S":
    model = PrognosisModelS().to(device)
else:
    raise NotImplementedError

train_dataset = SurvivalData(mode="train", transform=transform_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
val_dataset = SurvivalData(mode="val", transform=transform_val)
val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, num_workers=0, pin_memory=True)
test_dataset = SurvivalData(mode="test", transform=transform_val)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=0, pin_memory=True)

optimizer = torch.optim.Adam(model.parameters(), args.lr)

val_interval = 1
best_metric = -1
best_metric_test = -1
best_metric_epoch = -1
epoch_loss_values = []
metric_values = []
max_epochs = args.max_epoch

for epoch in range(max_epochs):
    print("-" * 10)
    print(f"epoch {epoch}/{max_epochs}")
    model.train()
    epoch_loss = 0
    step = 0

    risk_tmp = []
    time_tmp = []
    event_tmp = []

    for x_liver, x_lung, x_liver_1, x_lung_1, time, event in train_loader:
        step += 1

        x_liver = x_liver.to(device)
        x_lung = x_lung.to(device)
        x_liver_1 = x_liver_1.to(device)
        x_lung_1 = x_lung_1.to(device)
        time = time.to(device)
        event = event.to(device)
        out = model(x_liver, x_lung, x_liver_1, x_lung_1)

        risk_tmp.append(out.cpu().detach().squeeze())
        time_tmp.append(time.cpu().detach().squeeze())
        event_tmp.append(event.cpu().detach().squeeze())

        optimizer.zero_grad()
        loss = cox_ph_loss(log_h=out, durations=time, events=event)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_len = len(train_dataset) // train_loader.batch_size
        print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")

    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    c_index_value = concordance_index(torch.hstack(time_tmp), -np.exp(torch.hstack(risk_tmp)), torch.hstack(event_tmp))

    print(f"epoch {epoch} average loss: {epoch_loss:.4f}, c-index: {c_index_value:.4f}")

    if (epoch + 1) % val_interval == 0:
        model.eval()
        risk_tmp_val, time_tmp_val, event_tmp_val = [], [], []

        with torch.no_grad():
            for x_liver, x_lung, x_liver_1, x_lung_1, time, event in val_loader:
                x_liver = x_liver.to(device)
                x_lung = x_lung.to(device)
                x_liver_1 = x_liver_1.to(device)
                x_lung_1 = x_lung_1.to(device)
                time = time.to(device)
                event = event.to(device)
                out = model(x_liver, x_lung, x_liver_1, x_lung_1)

                risk_tmp_val.append(out.cpu().detach().squeeze())
                time_tmp_val.append(time.cpu().detach().squeeze())
                event_tmp_val.append(event.cpu().detach().squeeze())

            loss_val = cox_ph_loss(log_h=torch.hstack(risk_tmp_val),
                                   durations=torch.hstack(time_tmp_val),
                                   events=torch.hstack(event_tmp_val))

            c_index_val = concordance_index(torch.hstack(time_tmp_val),
                                            -np.exp(torch.hstack(risk_tmp_val)),
                                            torch.hstack(event_tmp_val))

        risk_tmp_test, time_tmp_test, event_tmp_test = [], [], []

        with torch.no_grad():
            for x_liver, x_lung, x_liver_1, x_lung_1, time, event in test_loader:
                x_liver = x_liver.to(device)
                x_lung = x_lung.to(device)
                x_liver_1 = x_liver_1.to(device)
                x_lung_1 = x_lung_1.to(device)
                time = time.to(device)
                event = event.to(device)
                out = model(x_liver, x_lung, x_liver_1, x_lung_1)

                risk_tmp_test.append(out.cpu().detach().squeeze())
                time_tmp_test.append(time.cpu().detach().squeeze())
                event_tmp_test.append(event.cpu().detach().squeeze())

            loss_test = cox_ph_loss(log_h=torch.hstack(risk_tmp_test),
                                    durations=torch.hstack(time_tmp_test),
                                    events=torch.hstack(event_tmp_test))

            c_index_test = concordance_index(torch.hstack(time_tmp_test),
                                             -np.exp(torch.hstack(risk_tmp_test)),
                                             torch.hstack(event_tmp_test))

            if c_index_val > best_metric:
                best_metric = c_index_val
                best_metric_epoch = epoch
            torch.save(model.state_dict(), os.path.join(save_ckpt_path, f"epoch_{epoch}_{loss_val:.4f}"
                                                                        f"_{c_index_val:.4f}_{loss_test:.4f}"
                                                                        f"_{c_index_test:.4f}.pth"))
            if c_index_test > best_metric_test:
                best_metric_test = c_index_test
            print(f"current epoch {epoch}, current loss {loss_val:.4f}, current val c-index {c_index_val:.4f}, "
                  f"best val c-index {best_metric:.4f} in epoch {best_metric_epoch}."
                  f" Current test c-index {c_index_test:.4f}, best test c-index {best_metric_test:.4f}")

