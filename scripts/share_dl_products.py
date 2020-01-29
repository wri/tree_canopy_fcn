from descarteslabs.catalog import Product, Band
#
# CONSTANTS
#
DEV=False
DEV_READERS=['email:brook.williams@gmail.com']
DEV_PRODUCT='wri:dev_1'

NGS=['org:ngs']
TREE_PRODUCT='wri:treecanopy'
ULU_PRODUCT='wri:ulu-india'
ULU_PREDS_PRODUCT='wri:ulu_predictions-india'


#
# METHODS
#
def add_readers(prod,readers):
    p=Product.get(prod)
    print(p)
    p.readers=list(set(p.readers+readers))
    p.save()
    status=p.update_related_objects_permissions(readers=readers)
    return p,status


#
# RUN
#
print('\n'*2)
print('='*50)

if DEV:
    p,dev_status=add_readers(DEV_PRODUCT,DEV_READERS)
    print(p.readers)
    if dev_status:
        dev_status.wait_for_completion()
    dev_status
    print(dev_status)
else:
    t_p,tree_status=add_readers(TREE_PRODUCT,NGS)
    u_p,ulu_status=add_readers(ULU_PRODUCT,NGS)
    # up_p,ulu_preds_status=add_readers(ULU_PREDS_PRODUCT,NGS)

    print(t_p.readers)
    if tree_status:
        tree_status.wait_for_completion()
    print(tree_status)

    print('-'*50)
    print(u_p.readers)
    if ulu_status:
        ulu_status.wait_for_completion()
    print(ulu_status)
    
    # print('-'*50)
    # print(up_p.readers)
    # if ulu_preds_status:
    #     ulu_preds_status.wait_for_completion()
    # print(ulu_preds_status)

print('='*50)
print('\n'*2)


