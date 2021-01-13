import matplotlib.pyplot as plt

def set_thai_font(font_family='tahoma'):
    plt.rcParams["font.family"] = font_family

def set_bigger_font(font_size=24):
    plt.rcParams.update({'font.size': font_size})
    
def auto_adjust():
    set_thai_font()
    set_bigger_font()