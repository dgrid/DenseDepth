#!/bin/bash
set -eu

USER_ID=$(id -u)
GROUP_ID=$(id -g)

# グループを作成する
if [ x"$GROUP_ID" != x"0" ]; then
    groupadd -g $GROUP_ID $USER
fi

# ユーザを作成する
if [ x"$USER_ID" != x"0" ]; then
    useradd -d /home/$USER -m -s /bin/bash -u $USER_ID -g $GROUP_ID $USER
fi

# パーミッションを元に戻す
sudo chmod u-s /usr/sbin/useradd
sudo chmod u-s /usr/sbin/groupadd

sudo chown -R ${USER}:${GROUP_ID} /home/${USER}
sudo chown -R ${USER}:${GROUP_ID} /workspace
echo "${USER}:USER" | sudo chpasswd

exec $@