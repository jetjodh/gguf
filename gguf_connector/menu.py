print("Please select a connector:\n1. llama.cpp\n2. ctransformers")
choice = input("Enter your choice (1 to 2): ")

if choice=="1":
    print("Connector: llama.cpp is selected!\nPlease select an interface:\n1. graphical\n2. command-line")
    option = input("Enter your choice (1 to 2): ")
    if option=="1":
        from .cpp import *
    elif option=="2":
        from .gpp import *
    else: print("Not a valid number.")
elif choice=="2":
    print("Connector: ctransformers is selected!\nPlease select an interface:\n1. graphical\n2. command-line")
    option = input("Enter your choice (1 to 2): ")
    if option=="1":
        from .c import *
    elif option=="2":
        from .g import *
    else: print("Not a valid number.")
else:
    print("Not a valid number.")