import os

a =os.path.dirname('/Users/liusen/Documents/sz/ATM/ATMServer/post_pic.py') + '/'
commond_remove = 'rm -rf {}*.jpg'.format(a)
print(commond_remove)