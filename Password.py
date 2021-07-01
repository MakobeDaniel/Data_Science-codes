import string
import random

class BasePasswordManager():
    def __init__(self, old_passwords):
        self.old_passwords = old_passwords
    def get_password(self):
        return self.old_passwords[-1]
    def is_correct(self, current_password):
        return current_password == self.get_password()

class PasswordManager(BasePasswordManager):
    def set_password(length):
        password_characters = string.ascii_letters + string.digits + string.punctuation
        new_password = []
        for i in range(length):
            new_password.append(random.choice(password_characters))
        print(''.join(new_password))

    def get_level(current_password):
        x = True
        while x:
            if current_password.isalpha() or current_password.isdigit():
                print("current_password is of level 0")
                break
            elif current_password.isalnum():
                print("current_password is of level 1")
                break
            else:
                print("current_password is of level 2")
                x = False
                break

    def get_level2(new_password):
        x = True
        while x:
            if (len(new_password)) < 6:
                print('new_passord not of accepted length')
                print('generate another password of length > 6')
                break
            elif new_password.isalpha() or new_password.isdigit():
                print('new_password is of level 0')
                break
            elif new_password.isalnum():
                print('new_password is of level 1')
                break
            else:
                print('new_password is of level 2')
                print('Password change is SUCCESSFUL')
                x = False
                break

bpm = BasePasswordManager(['224466','zxywu','AbCdEquality', 'Jupyterishere','GameChanger123'])
print(bpm.get_password())
print(bpm.is_correct('GameChanger123'))
current_password_level = PasswordManager.get_level((input(('Enter the Current_password above'))
new_password = PasswordManager.set_password(8)
new_password_level = PasswordManager.get_level2(input('Enter the newpass generated above'))