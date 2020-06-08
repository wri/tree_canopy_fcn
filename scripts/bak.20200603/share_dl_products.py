from descarteslabs.catalog import Product, Band
from time import sleep
#
# CONSTANTS
#
DEV=False
DEV_READERS=['org:wri']
DEV_OWNERS=['org:wri']
DEV_PRODUCT='wri:dev_1'

READERS=['org:wri','org:ngs']
OWNERS=['org:wri']

TREE_PRODUCT='wri:treecanopy'
ULU_PRODUCT='wri:ulu-india'

SAVED='SAVED'
WAIT=10


#
# METHODS
#
def update_permissions(prod,readers=None,owners=None,update=False):
    p=Product.get(prod)
    print(p)
    try:
        
        if update:
            if readers:
                readers=list(set(readers+p.readers))
            if owners:
                owners=list(set(owners+p.owners))
        if readers:
            p.readers=readers
        if owners:
            p.owners=owners
        p.save()
        
        if SAVED in str(p.state):
            print('waiting for product to save...')
            sleep(WAIT)
        if SAVED in str(p.state):
            status=p.update_related_objects_permissions(inherit=True)
            return p, status
        else:
            print('WARNING: saving product delayed or failed')
            return p, None
        
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

    p,dev_status=update_permissions(DEV_PRODUCT,readers=DEV_READERS,owners=DEV_OWNERS)
    if dev_status:
        print(f'{DEV_PRODUCT}:',p.readers)
        dev_status.wait_for_completion()
    dev_status
    print(dev_status)

else:

    t_p,tree_status=update_permissions(TREE_PRODUCT,readers=READERS,owners=OWNERS)
    u_p,ulu_status=update_permissions(ULU_PRODUCT,readers=READERS,owners=OWNERS)

    if t_p:
        if tree_status:
            print(f'{TREE_PRODUCT}:',t_p.readers)
            tree_status.wait_for_completion()
        print(tree_status)

    if u_p:
        print('-'*50)
        if ulu_status:
            print(f'{ULU_PRODUCT}:',u_p.readers)
            ulu_status.wait_for_completion()
        print(ulu_status)

print('='*50)
print('\n'*2)


