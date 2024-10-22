from matplotlib.widgets import Slider
import matplotlib.pyplot as plt

def test():
    plt.figure(2, figsize=[5, 1])
    ax_aperture = plt.axes([0.25, 0.5, 0.65, 0.3], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_aperture, 'Aperture Radius', 0.0, 10.0, valinit=5.0)
    slider.on_changed(update_slider)
    plt.show()

def update_slider(val):
    print(val)



if __name__ == '__main__':
    test()