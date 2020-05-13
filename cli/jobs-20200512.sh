# jobs-20200512.sh


# 1. green-naip
echo
echo 
echo greenspace_naip
date
echo
echo
torch_kit train greenspace_naip --poweroff f --dry_run f --dev f


# 2. all-naip
echo
echo
echo green_bu_naip
date 
echo
echo
torch_kit train green_bu_naip --poweroff f --dry_run f --dev f


# 1. green
echo
echo
echo greenspace
date
echo
echo
torch_kit train greenspace --poweroff f --dry_run f --dev f


# 2. all
echo
echo
echo green_bu
date
echo
echo
torch_kit train green_bu --poweroff f --dry_run f --dev f


# 3. bu
echo
echo
echo bu
date
echo
echo
torch_kit train bu --dry_run f --dev f


# ---- DONE
echo
echo
echo
echo
date
echo
echo
echo
echo