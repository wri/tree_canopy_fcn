from descarteslabs.catalog import Product, Band
#
# CONSTANTS
#
DEV=False
DEV_READERS=['org:wri']
DEV_OWNERS=[
    'org:wri',
    'user:6d27def1bb7fb0138933a4ee2e33cce9f5af999a',
    'user:9abfd2e5a5981bbd91d426119af32a916473d427',
    'user:9dcf8e7b0b88050ff98661e91051550313f0c232']
DEV_PRODUCT='wri:dev_1'

# NGS=['org:ngs']
# OWNERS=['user:6d27def1bb7fb0138933a4ee2e33cce9f5af999a','user:9abfd2e5a5981bbd91d426119af32a916473d427']
READERS=['org:wri','org:ngs']
OWNERS=[
    'org:wri',
    'user:6d27def1bb7fb0138933a4ee2e33cce9f5af999a',
    'user:9abfd2e5a5981bbd91d426119af32a916473d427',
    'user:9dcf8e7b0b88050ff98661e91051550313f0c232']
TREE_PRODUCT='wri:treecanopy'
ULU_PRODUCT='wri:ulu-india'
ULU_PREDS_PRODUCT='wri:ulu_predictions-india'


#
# METHODS
#
def add_readers(prod,readers,owners=None,update=True):
    p=Product.get(prod)
    print(p)
    try:
        if update:
            if readers:
                readers=list(set(readers+p.readers))
            if owners:
                owners=list(set(owners+p.owners))
        p.readers=readers
        if owners:
            p.owners=owners
        p.save()
        status=p.update_related_objects_permissions(owners=owners,readers=readers)
        return p,status
    except Exception as e:
        print('ERROR:',e)
        print()
        return False, False


#
# RUN
#
print('\n'*2)
print('='*50)

if DEV:
    p,dev_status=add_readers(DEV_PRODUCT,DEV_READERS,DEV_OWNERS,False)
    if dev_status:
        print(p.readers)
        dev_status.wait_for_completion()
    dev_status
    print(dev_status)
else:

    # t_p,tree_status=add_readers(TREE_PRODUCT,NGS,owners=OWNERS)
    u_p,ulu_status=add_readers(ULU_PRODUCT,READERS,OWNERS,False)
    # up_p,ulu_preds_status=add_readers(ULU_PREDS_PRODUCT,NGS)

    # if t_p:
    #     if tree_status:
    #         print(t_p.readers)
    #         tree_status.wait_for_completion()
    #     print(tree_status)

    if u_p:
        print('-'*50)
        if ulu_status:
            print(u_p.readers)
            ulu_status.wait_for_completion()
        print(ulu_status)
    
    # print('-'*50)
    # print(up_p.readers)
    # if ulu_preds_status:
    #     ulu_preds_status.wait_for_completion()
    # print(ulu_preds_status)

print('='*50)
print('\n'*2)


