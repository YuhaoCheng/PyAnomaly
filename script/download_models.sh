#! /bin/bash
mkdir ../pretrained_model
cd ../pretrained_model
mkdir flownet2
cd flownet2
wget --verbose --continue --timestamping https://drive.google.com/file/d/1hF8vS6YeHkx3j2pfCeQqqZGwA_PJq_Da/view?usp=sharing

cd ..
mkdir litflownet
cd litflownet
wget --verbose --continue --timestamping http://content.sniklaus.com/github/pytorch-liteflownet/network-default.pytorch
wget --verbose --continue --timestamping http://content.sniklaus.com/github/pytorch-liteflownet/network-kitti.pytorch
wget --verbose --continue --timestamping http://content.sniklaus.com/github/pytorch-liteflownet/network-sintel.pytorch
