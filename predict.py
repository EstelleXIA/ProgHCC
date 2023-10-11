import os
import SimpleITK as sitk
import torchvision.transforms as transforms
from model import *
from monai.transforms import NormalizeIntensity, Resize
import argparse
import numpy as np


def list_of_strings(arg):
    return arg.split(',')


parser = argparse.ArgumentParser()
parser.add_argument('--base_liver', type=str, help='base directory for liver', default="LIVER/PATH/")
parser.add_argument('--base_lung', type=str, help='base directory for lung', default="LUNG/PATH/")
parser.add_argument('--baseline_liver', type=list_of_strings, help='a list of filename of the baseline liver slices',
                    default="slice_1.nii.gz,slice_2.nii.gz,slice_3.nii.gz")
parser.add_argument('--baseline_lung', type=list, help='a list of filename of the baseline lung slices',
                    default="slice_1.nii.gz,slice_2.nii.gz,slice_3.nii.gz")
parser.add_argument('--followup_liver', type=list, help='a list of filename of the followup liver slices',
                    default="slice_1.nii.gz,slice_2.nii.gz,slice_3.nii.gz")
parser.add_argument('--followup_lung', type=list, help='a list of filename of the followup lung slices',
                    default="slice_1.nii.gz,slice_2.nii.gz,slice_3.nii.gz")
parser.add_argument('--model', type=str, help='Rad-D or Rad-S', default="Rad-D")
parser.add_argument('--Differentiation', type=int,
                    help='Histological type, 0 for high differentiation, 1 for moderately differentiation,'
                         ' 2 for low differentiation, and 3 for undifferentiation', default=1)
parser.add_argument('--NASH', type=int, help='non-alcoholic steatohepatitis or non-alcoholic steatohepatitis,'
                                             ' 0 for no, and 1 for yes', default=0)
parser.add_argument('--Surgery', type=int, help='0 for non-surgery and 1 for surgery', default=0)
parser.add_argument('--PVT', type=int, help='partial or complete portal vein tumor thrombosis,'
                                            ' 0 for no, and 1 for yes', default=0)
parser.add_argument('--EBRT', type=int, help='external beam radiation therapy,'
                                             ' 0 for no, and 1 for yes', default=0)
parser.add_argument('--TACE', type=int, help='transarterial embolization or transarterial chemoembolization,'
                                             ' 0 for no, and 1 for yes', default=0)
parser.add_argument('--RFAMWA', type=int, help='radiofrequency ablation or microwave ablation,'
                                               ' 0 for no, and 1 for yes', default=0)

args = parser.parse_args()


transform = transforms.Compose([
    Resize(spatial_size=(224, 224)),
    NormalizeIntensity()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_liver, x_lung, x_liver_1, x_lung_1 = [], [], [], []

for file in args.baseline_liver:
    img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(args.base_liver, file)))
    if transform:
        img = transform(img)
    x_liver.append(img[0])
x_liver = torch.stack(x_liver).unsqueeze(0)

for file in args.baseline_lung:
    img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(args.base_lung, file)))
    if transform:
        img = transform(img)
    x_lung.append(img[0])
x_lung = torch.stack(x_lung).unsqueeze(0)


if args.model == "Rad-D":
    for file in args.followup_liver:
        img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(args.base_liver, file)))
        if transform:
            img = transform(img)
        x_liver_1.append(img[0])
    x_liver_1 = torch.stack(x_liver_1).unsqueeze(0)
    for file in args.followup_lung:
        img = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(args.base_lung, file)))
        if transform:
            img = transform(img)
        x_lung_1.append(img[0])
    x_lung_1 = torch.stack(x_lung_1).unsqueeze(0)
    model = PrognosisModelD().to(device)
    model.load_state_dict(torch.load("rad-D-pretrained.pth"))
    out = model(x_liver, x_lung, x_liver_1, x_lung_1)
elif args.model == "Rad-S":
    model = PrognosisModelS().to(device)
    model.load_state_dict(torch.load("rad-S-pretrained.pth"))
    out = model(x_liver, x_lung)
else:
    raise NotImplementedError

radiology_score = out
clinical_score = 0.3747*args.Differentiation+0.1593*args.NASH-0.1801*args.Sugery +\
                 0.6732*args.PVT-0.8235*args.EBRT+0.6482*args.TACE-0.4497*args.RFAMWA
final_risk = np.log(9.883337*radiology_score + 0.530013 * clinical_score)

print(f"The final risk predicted by {args.model} is {final_risk}.")
print("The risk higher than 0.66 is considered high-risk.")
