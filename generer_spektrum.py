import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from numba import njit
from time import time
from scipy.interpolate import interp1d
from IPython.display import clear_output
from scipy.interpolate import UnivariateSpline
import os


def linear_function(x, *args):
    a, b = args
    return a*x + b

def quadratic_funtion(x, *args):
    a,b,c = args
    return a*x**2 + b *x + c

def Laboratory_data(filename):
    # Load data, allowing for space or tab as delimiters
    data = np.genfromtxt(filename, delimiter=None, dtype=str)  # Read as strings to handle commas
    data = np.char.replace(data, ',', '.')  # Replace commas with dots
    data = data.astype(float)  # Convert to float after replacement
    
    # Extract columns
    x_values = data[:, 0] / 10  # Scale x-axis
    y_values = data[:, 1]
    return x_values, y_values

def get_slope(image_path):
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError("Could not read the image. Check the file path.")
    
    
    rows, cols = img.shape
    
    max_intensities = []
    x_coordinates = []
    y_coordinates = []
    
    
    
    for row in img:
        max_intensities.append(row[np.argmax(row)])
    
    for i, row in enumerate(img):
        if row[np.argmax(row)]>= max(max_intensities)/10:
            y_coordinates.append(rows-i)
            x_coordinates.append(np.argmax(row))
        
    fun_para, _ = curve_fit(linear_function, x_coordinates, y_coordinates, [1,1])
    
    # Checking fit
    # x_axis = np.arange(0,cols)
    # plt.plot(x_axis, fun_para[0]*x_axis + fun_para[1])
    
    normal_slope = -1/fun_para[0]
    
    return normal_slope


def generate_spectrum(image_path, threshold, normal_slope):
    # Load the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img_color = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Could not read the image. Check the file path.")
    
    rows, cols = img.shape
    
    some_row = len(img)//2
    bottom_total_color = np.array([0,0,0]) #Blue Green Red
    top_total_color = np.array([0,0,0])
    for row in range(len(img)):
        if row > some_row:
            bottom_total_color += np.array(img_color[row][np.argmax(img[row])])
        else:
            top_total_color += np.array(img_color[row][np.argmax(img[row])])

    amount_of_blue_at_bottom = bottom_total_color[0]/bottom_total_color[2]
    amount_of_blue_at_top = top_total_color[0]/top_total_color[2]
    is_blue_at_the_top = amount_of_blue_at_bottom < amount_of_blue_at_top
    
    if is_blue_at_the_top:
        new_img = [[pixel for pixel in row[::-1]] for row in img[::-1]]
        img = np.array(new_img)
    
    
    @njit # for koden til at køre meget hurtigere :^)
    def get_spectrum():
        b = rows-(normal_slope*cols)
        spectrum = []
        print(int(b))
        for offset in range(int(b)):
            intensity_of_current_line = 0
            for x in range(cols):
                y = rows-int(normal_slope*x + offset)
                if 0<=y<rows:
                    intensity = img[y][x]
                    if intensity > threshold: # fjerner sorte pixels
                        intensity_of_current_line += intensity 
            spectrum.append(intensity_of_current_line)
        return spectrum
    
    spectrum = get_spectrum()

    return spectrum

def calibrering(spectrum ,peaks, tabular_values):
    peak_location = []
    for i in peaks:
        location = np.argmin(spectrum[i-30:i+30])+i-30
        peak_location.append(location)
    
    calibration_parameters, _ = curve_fit(quadratic_funtion, peak_location, tabular_values, [1,1,1])
    
    calibrated_x_axis = quadratic_funtion(np.arange(0, len(spectrum), 1), *calibration_parameters)
        
    return calibrated_x_axis, peak_location

def get_average_spectrum(image_path, frame_number, normal_slope ):

    # normal_slope = get_slope(f"{image_path}{n_images//2}.jpg")
    spectra = []
    for i in range(frame_number):
        print("Generating spectrum for frame", i)
        spectra.append(np.array(generate_spectrum(f"{image_path}{i}.jpg", threshold = 0.01, normal_slope = normal_slope)))
    clear_output()

    return sum(spectra)

def extract_frame(video_path, output_folder,output_image_path, antal_billeder):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return
    
    frames = []
    frame_count = 0
    frame_index = 0  # Counter to track frames

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # Stop if no more frames
        
        # Take every `frame_interval`-th frame
        if frame_index == antal_billeder:
            frames.append(frame)
        
        frame_index += 1
    
    cap.release()
    
    if not frames:
        print("No frames were extracted from the video.")
        return
    
    print(f"Total frames extracted: {len(frames)}")
    
    # Initialize the stacked image with zeros (same shape as one frame)
    stacked_image = np.zeros_like(frames[0], dtype=np.float32)
    
    # Add the frames together (pixel-by-pixel summing)
    for frame in frames:
        stacked_image += frame.astype(np.float32)  # Add pixel values
    
    # Normalize the stacked image to prevent overflow (clipping)
    stacked_image = np.clip(stacked_image, 0, 255).astype(np.uint8)

    output_image = os.path.join(output_folder, output_image_path)
    # Save the result
    # cv2.imwrite(output_image_path, stacked_image)
    cv2.imwrite(output_image, stacked_image)
    print(f"Stacked image saved as {output_image_path}")

def plot_dat_file(filename, transition_lines=None):
    # Load data, allowing for space or tab as delimiters
    data = np.genfromtxt(filename, delimiter=None, dtype=str)  # Read as strings to handle commas
    # print(data)
    data = np.char.replace(data, ',', '.')  # Replace commas with dots
    data = data.astype(float)  # Convert to float after replacement
    
    # Extract columns
    x_values = data[:, 0] / 10  # Scale x-axis
    y_values = data[:, 1]

    
    return x_values,y_values


def generer_spektrum_fra_videoer(video_1,video_2,antal_billeder, roter=False):
    
    # blå_billeder = r'C:\Users\nebse\Desktop\Jupiter\Spectroscopy\Colab_scripts\spectrum_blue_frame'
    # rød_billeder = r'C:\Users\nebse\Desktop\Jupiter\Spectroscopy\Colab_scripts\spectrum_red_frame'
    blå_billeder = "blue_billede"
    rød_billeder = "red_billede"
    mappe = "Billeder"
    figur_mappe = "Grafer"
    if not os.path.exists(figur_mappe):
        os.makedirs(figur_mappe)

    # try:
    for i in range(antal_billeder):
        extract_frame(video_1,mappe,f'{blå_billeder}{i}.jpg', antal_billeder=i)
        extract_frame(video_2,mappe,f'{rød_billeder}{i}.jpg', antal_billeder=i) 

    clear_output()

    # blå_normal_slope = get_slope(rf"{mappe}\{blå_billeder}{antal_billeder//2}.jpg")
    # rød_normal_slope = get_slope(rf"{mappe}\{rød_billeder}{antal_billeder//2}.jpg")
    
    location = "/content/Billeder/"
    blå_normal_slope = get_slope(f"{location}{blå_billeder}{antal_billeder//2}.jpg")
    rød_normal_slope = get_slope(f"{location}{blå_billeder}{antal_billeder//2}.jpg")
    

    # spectre_blå = get_average_spectrum(rf"{mappe}\{blå_billeder}",antal_billeder,blå_normal_slope)
    # spectre_rød = get_average_spectrum(rf"{mappe}\{rød_billeder}",antal_billeder,rød_normal_slope)
    
    spectre_blå = get_average_spectrum(f"{location}{blå_billeder}",antal_billeder,blå_normal_slope)
    spectre_rød = get_average_spectrum(f"{location}{rød_billeder}",antal_billeder,rød_normal_slope)


    fig, ax = plt.subplots(2,1,figsize = (12,6))

    ax[0].plot(spectre_blå, linewidth = 1, color = "#1f77b4")
    ax[1].plot(spectre_rød, linewidth = 1, color = "#d62728")

    ax[0].text(0.05, 0.90, "Blå del af spektret", transform=ax[0].transAxes,  
        fontsize=12, verticalalignment='top', horizontalalignment='left',  
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))
    ax[1].text(0.95, 0.90, "Rød del af spektret", transform=ax[1].transAxes,  
        fontsize=12, verticalalignment='top', horizontalalignment='right',  
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))

    ax[1].set_xlabel("Pixel")

    for i in range(2):
            ax[i].set_xticks(np.arange(0,len(spectre_blå),250))
            ax[i].set_yticklabels([])
            ax[i].set_yticks([])
    figure_path = os.path.join(figur_mappe, "Delte spektrum.png")
    fig.savefig(figure_path)
    clear_output()
    plt.show()

    return spectre_blå, spectre_rød

    # except Exception:
    #     print("Venligst upload en video")
    #     return None, None

  
# Markere fælles lokation på billderne
def fælles_punkt(spektrum_blå, spektrum_rød, blå_punkt, rød_punkt, samle ):
    
    minima_blå = np.argmin(spektrum_blå[blå_punkt-20:blå_punkt+20])+blå_punkt-20
    minima_rød = np.argmin(spektrum_rød[rød_punkt-20:rød_punkt+20])+rød_punkt-20
    
    figur_mappe = "Grafer"
    
    offset = minima_blå-minima_rød
    Total_spectra = []
    total_visuable_spectra = []
    average_situation = True
    
    for i in range(len(spektrum_rød)+offset):
        if i <= minima_blå:
            Total_spectra.append(spektrum_blå[i])
        elif average_situation == True and min(spektrum_blå[i:i+30]) > 0 :
            Total_spectra.append((spektrum_blå[i]+spektrum_rød[i-offset])/2)
        else:
            average_situation = False
            Total_spectra.append(spektrum_rød[i-offset])
    
    for i in Total_spectra:
        if i > 0:
            total_visuable_spectra.append(i/max(Total_spectra))
   
    if samle == False:
        fig, ax = plt.subplots(2,1,figsize = (12,6))

        ax[0].plot(spektrum_blå, linewidth = 1, color = "#1f77b4")
        ax[1].plot(spektrum_rød, linewidth = 1, color = "#d62728")

        ax[0].plot([minima_blå,minima_blå],[min(spektrum_blå),max(spektrum_blå)],ls = '--',color = "black")
        ax[1].plot([minima_rød,minima_rød],[min(spektrum_rød),max(spektrum_rød)],ls = '--',color = "black")
        
        ax[1].set_xlabel("Pixel")
        
        for i in range(2):
            ax[i].set_xticks(np.arange(0,len(spektrum_blå),250))
            ax[i].set_yticklabels([])
            ax[i].set_yticks([])
        
        figure_path = os.path.join(figur_mappe, "Samlepunkt.png")
        fig.savefig(figure_path)
    
    if samle:
        
        fig, ax = plt.subplots(1,1,figsize = (12,4))
        

        ax.plot(total_visuable_spectra, linewidth = 1, color = "black")
        ax.set_title("Hele Spektret")
        ax.set_xlabel("Pixel")
        ax.set_xticks(np.arange(0,3001,250))
        ax.set_yticklabels([])
        ax.set_yticks([])
        
        figure_path = os.path.join(figur_mappe, "Spektrum.png")
        fig.savefig(figure_path)
        
        return total_visuable_spectra

# calibreing
def kalibreret_graf(spektrum, absorptions_linjer, tabel_værdier, kalibrering):
    
    x_axis, peaks = calibrering(spektrum, absorptions_linjer, tabel_værdier)
    clear_output()
    if kalibrering == False:
        fig, ax = plt.subplots(2,1, figsize = (12,6))
        
        # x, y = plot_dat_file("/content/drive/MyDrive/Jupiters_måner/Scripts/g2v_human.dat")
        x, y = plot_dat_file("/content/spektroskopi/g2v_tabel.dat")
        # x, y = plot_dat_file(r"C:\Users\nebse\Desktop\Jupiter\Spectroscopy\solen\g2v_human.dat")
        
        hydrogen_beta_guess = 486.1
        hydrogen_alpha_guess = 656.3
        Na_guess = 589
        
        absorption_sun = [(r'H$_\beta$',hydrogen_beta_guess),(r'H$_\alpha$',hydrogen_alpha_guess),('Na',Na_guess)]
        
        xs = []
        ys = []

        for i in range(len(x)):
            if 400<=x[i]<=700:
                xs.append(x[i])
                ys.append(y[i])
        

        for peak in absorption_sun:
            for j in range(len(xs)):
                if xs[j]>=peak[1]-3 and xs[j]<=peak[1]+3:
                    min_punkt = np.min(ys[j-10:j+10])
                    ax[1].plot([peak[1],peak[1]],[0.6,min_punkt],ls = '--', color = "#d62728",linewidth = 0.75)
                    ax[1].text(peak[1]+1, np.mean([min_punkt,0.6]), peak[0], rotation=0, verticalalignment='bottom', fontsize=10, color='#d62728')
        
        ax[1].plot(xs,ys, linewidth = 1, color = "black")
        # ax[1].set_title("Solens Spektrum")
        ax[1].set_xlabel("Bølgelængde (nm)")
        ax[1].set_xticks([400,425, 450,475, 500, 525, 550,575, 600,625, 650,675, 700])
       
        ax[0].text(0.05, 0.90, "Vores Spektrum", transform=ax[0].transAxes,  
        fontsize=12, verticalalignment='top', horizontalalignment='left',  
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))
        ax[1].text(0.95, 0.90, "Solens Spektrum", transform=ax[1].transAxes,  
        fontsize=12, verticalalignment='top', horizontalalignment='right',  
        bbox=dict(facecolor='white', alpha=0.7, edgecolor='black'))
        
        for peak in peaks:
            ax[0].plot([peak,peak],[min(spektrum),max(spektrum)],'--r', linewidth = 1)    
        ax[0].plot(spektrum, linewidth = 1, color = "black") 
        ax[0].set_xticks([0,250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000])
        # ax[0].set_xlabel("Pixel")
        
        # Fælles plot elementer:
        for i in range(2):
            ax[i].set_yticklabels([])
            ax[i].set_yticks([])
            
        figur_mappe = "Grafer"
        figure_path = os.path.join(figur_mappe, "Kalibreringspunkter.png")
        fig.savefig(figure_path)

        return x_axis, spektrum
    
    elif kalibrering:
        fig, ax = plt.subplots(1,1, figsize = (12,4))
        
        ax.plot(x_axis,spektrum, linewidth = 1, color = "black")
        ax.set_title("Spektrum (kalibreret)")
        ax.set_xlabel("Bølgelængde (nm)")
        ax.set_xticks([400,425, 450,475, 500, 525, 550,575, 600,625, 650,675, 700])
        ax.set_yticklabels([])
        ax.set_yticks([])
        
        figur_mappe = "Grafer"
        figure_path = os.path.join(figur_mappe, "Kalibreret graf.png")
        fig.savefig(figure_path)
        
        return x_axis, spektrum

# Fit til en funktion
def fit(spektrum, afvigelse,korrekt = True):
    def create_smooth_spline(y, smoothing_factor=1, num_points=len(spektrum)):
        x = np.arange(len(y))  # Generate x values as indices
        spline = UnivariateSpline(x, y, s=smoothing_factor)
        x_smooth = np.linspace(min(x), max(x), num_points)
        y_smooth = spline(x_smooth)
        
        return  y_smooth
    

    smoothing_factor = 0.1 # Adjust this for more or less smoothness

    y_smooth = create_smooth_spline(spektrum, smoothing_factor)

    def best_normaliser(spectre, spline,smoothing_factor,afvigelse, count = 0):
        difference = 0
        print(smoothing_factor)
        for i in range(len(spectre)):
            difference += abs(spectre[i]-spline[i])
    
            
        # if difference > len(spline)*afvigelse/100:
        if difference > len(spline)*afvigelse/1000:
            print('difference reach threshold')
            print(f'difference calculated:{difference}')
            new_list = spline
            return new_list
        
        # print(f'count: {count}, difference: {difference}' )
        count+=1
        print(f'number of recursions: {count} number of difference: {difference}')

        smoothing_factor *= 1.1
        y_smooth_new = create_smooth_spline(spectre, smoothing_factor)
        
        return best_normaliser(spectre, y_smooth_new,smoothing_factor,afvigelse, count)

    spline = best_normaliser(spektrum,y_smooth, smoothing_factor,afvigelse)
    
    if korrekt:
        clear_output()
        fig, ax = plt.subplots(1,1, figsize = (12,4))
        
        ax.plot(spektrum, linewidth = 1, color = "black", label = "Spektrum")
        ax.plot(spline,linewidth = 2, color = "r", label = "Blød graf")
        ax.set_title("Blød graf over Spektret")
        ax.legend()
        ax.set_xticks([])
        ax.set_yticklabels([])
        ax.set_yticks([])
        return spline
    
    if korrekt == False :
        clear_output()
        return spline


def normaliser(kalibreret_x, spektrum, afvigelse):
    
    spline = fit(spektrum, afvigelse, korrekt = False)
    
    fig, ax = plt.subplots(2,1,figsize = (12,6))
    
    index_location_of_shit = []
    local_minimums = []
    
    normalised_spectre_try = []
    
    for i in range(0,len(spektrum)):
        if spektrum[i]>spline[i]:
            normalised_spectre_try.append((spektrum[i]-spline[i])**(2)*0.5+0.5)
        else :
            normalised_spectre_try.append(-(spektrum[i]-spline[i])**(2)+0.5)
            
            
    for i in range(len(spektrum)):
        if kalibreret_x[i-40]<471<kalibreret_x[i+40]:
            index_location_of_shit.append(i)
            local_minimums.append(np.min(normalised_spectre_try[i-20:i+20]))
    
    min_location = np.min(local_minimums)
    the_index_location = index_location_of_shit[np.where(local_minimums == min_location)[0][0]]
    
    
    
    afvigelses_forhold = 12
    
    spline1 = fit(spektrum[:the_index_location], afvigelse, korrekt = False)
    spline2 = fit(spektrum[the_index_location:], afvigelse*afvigelses_forhold, korrekt = False)
    
    # ax[0].plot(kalibreret_x[:the_index_location],spline1)
    # ax[0].plot(kalibreret_x[the_index_location:],spline2)
    # print(the_index_location)
    
    total_spline = list(spline1)+list(spline2)
    
    normalised_spectre = []
    
    for i in range(0,len(spektrum)):
        if spektrum[i]>total_spline[i]:
            normalised_spectre.append((spektrum[i]-total_spline[i])**(2)*0.5+0.5)
        else :
            normalised_spectre.append(-(spektrum[i]-total_spline[i])**(2)+0.5)
    
    # clear_output()
    ax[0].plot(kalibreret_x,spektrum, linewidth = 1, color = "black", label = "Spektrum")
    ax[0].plot(kalibreret_x,total_spline,linewidth = 2, color = "r", label = "Blød graf")
    ax[0].set_title("Blød graf over Spektret")
    ax[0].legend()
    ax[0].set_xticks([])
    ax[0].set_yticklabels([])
    ax[0].set_yticks([])
    
    ax[1].plot(kalibreret_x, normalised_spectre, color = "black",linewidth = 1)
    ax[1].set_title("Normaliseret Spektrum (Spektrum/blød graf)")
    ax[1].set_yticklabels([])
    ax[1].set_yticks([])
    ax[1].set_xlabel("Bølgelængde (nm)")
    ax[1].set_xticks([400,425, 450,475, 500, 525, 550,575, 600,625, 650,675, 700])
    
    figur_mappe = "Grafer"
    figure_path = os.path.join(figur_mappe, "Normaliserings graf.png")
    fig.savefig(figure_path)
        
    return normalised_spectre


# Sammenlign
def sammenlign_med_solen(kalibreret_x,normalisered_spektrum):
    # Sol Spectre 

    x, y = plot_dat_file("/content/spektroskopi/g2v_normaliseret.dat")
    # x, y = plot_dat_file(r"C:\Users\nebse\Desktop\Jupiter\Spectroscopy\solen\g2v - absorptionslinjer 2.dat")
    fig, ax = plt.subplots(1,1,figsize = (12,6))

    Carbon_guess = 430.5
    hydrogen_beta_guess = 486.1
    hydrogen_alpha_guess = 656.3
    Na_guess = 589
    Mg_guess = 517.3
    Fe_guess = 527
    
    offset = np.min(y)-1.1
    
    
    
    sol_variation = np.max(y)-np.min(y)
    planet_variation = np.max(normalisered_spektrum)-np.min(normalisered_spektrum)
    
    
    relativ_variation = sol_variation/planet_variation
    

    skaleret_spektrum = np.array(normalisered_spektrum)*relativ_variation
    offset_skalering = normalisered_spektrum[0]-skaleret_spektrum[0]
    plaseret_skaleret_spektrum = skaleret_spektrum + offset_skalering
    
   
    absorption_sun = [('Ca',Carbon_guess),(r'H$_\beta$',hydrogen_beta_guess),(r'H$_\alpha$',hydrogen_alpha_guess),('Na',Na_guess),('Mg',Mg_guess),('Fe',Fe_guess)]
    
    for i in absorption_sun:
        for j in range(len(kalibreret_x)):
            if kalibreret_x[j]>=i[1]-1 and kalibreret_x[j]<=i[1]+1:
                min_punkt = np.min(plaseret_skaleret_spektrum[j-2:j+2])
                ax.plot([i[1],i[1]],[offset,min_punkt],ls = '--',linewidth = 1.1, color = "#d62728")
                if i[0] == 'Mg':
                    ax.text(i[1]-10, 0.1, i[0], rotation=0, verticalalignment='bottom', fontsize=12, color='#d62728')
                else:
                    ax.text(i[1]+1, 0.1, i[0], rotation=0, verticalalignment='bottom', fontsize=12, color='#d62728')
                break
    #d62728
    normalisered_sol = []
    x_axis = []
    for i in range(len(y)):
        if kalibreret_x[0]<=x[i]<=kalibreret_x[-1]:
            x_axis.append(x[i])
            if y[i]>1:
                normalisered_sol.append(y[i]*0.97-1)
            else: 
                normalisered_sol.append(y[i]-1)
    
    # clear_output()
    ax.plot(kalibreret_x,plaseret_skaleret_spektrum, linewidth = 1, color = 'k', label = "Vores Spektrum")
    ax.plot(x_axis,normalisered_sol, linewidth = 1, color = "#1f77b4", label = "Solens Spektrum")
    
    ax.legend()
    ax.set_xlabel("Bølgelængde (nm)")
    ax.set_xticks([400,425, 450,475, 500, 525, 550,575, 600,625, 650,675, 700])
    ax.set_yticklabels([])
    ax.set_yticks([])

    figur_mappe = "Grafer"
    figure_path = os.path.join(figur_mappe, "Sammenligning med solen.png")
    fig.savefig(figure_path)

def Find_molekyler(kalibreret_x, normalisered_spektrum, molekyle = None):   
    x, y = plot_dat_file("/content/spektroskopi/g2v_normaliseret.dat")
    # x, y = plot_dat_file(r"C:\Users\nebse\Desktop\Jupiter\Spectroscopy\solen\g2v - absorptionslinjer 2.dat")
    fig, ax = plt.subplots(1,1,figsize = (12,6))

    Carbon_guess = 430.5
    hydrogen_beta_guess = 486.1
    hydrogen_alpha_guess = 656.3
    Na_guess = 589
    Mg_guess = 517.3
    Fe_guess = 527
    
    offset = np.min(y)-1.1
    
    sol_variation = np.max(y)-np.min(y)
    planet_variation = np.max(normalisered_spektrum)-np.min(normalisered_spektrum)
    relativ_variation = sol_variation/planet_variation
    
    skaleret_spektrum = np.array(normalisered_spektrum)*relativ_variation
    offset_skalering = normalisered_spektrum[0]-skaleret_spektrum[0]
    plaseret_skaleret_spektrum = skaleret_spektrum + offset_skalering
    
   
    absorption_sun = [('Ca',Carbon_guess),(r'H$_\beta$',hydrogen_beta_guess),(r'H$_\alpha$',hydrogen_alpha_guess),('Na',Na_guess),('Mg',Mg_guess),('Fe',Fe_guess)]
    
    
    molekyle_navne = ['Kuldioxid','Svovldioxid', 'Oxygen', 'Methan']
    respektive_absorptionslinjer = [(480,510,660),(420,540,600),(577,630,687),(619,665,700)]
    
    for i in range(len(molekyle_navne)):
        if molekyle_navne[i] == molekyle:
            absorptionslinjer = respektive_absorptionslinjer[i]
            
    
    
    for bølgelængde in absorptionslinjer:
        for index in range(len(kalibreret_x)):
            if kalibreret_x[index]>=bølgelængde-1 and kalibreret_x[index]<=bølgelængde+1:
                min_punkt = np.min(plaseret_skaleret_spektrum[index-1:index+1])
                if bølgelængde == absorptionslinjer[0]:
                    ax.plot([bølgelængde,bølgelængde],[offset,min_punkt],ls = "--", color = "#ff7f0e",linewidth = 2,label = molekyle)
                else:
                    ax.plot([bølgelængde,bølgelængde],[offset,min_punkt],ls = "--",linewidth = 2, color = "#ff7f0e")
                break
    
    
    for i in absorption_sun:
        for j in range(len(kalibreret_x)):
            if kalibreret_x[j]>=i[1]-1 and kalibreret_x[j]<=i[1]+1:
                min_punkt = np.min(plaseret_skaleret_spektrum[j-2:j+2])
                if i[0] == 'Ca':
                    ax.plot([i[1],i[1]],[offset,min_punkt],ls = ':',linewidth = 1.1, color = "gray", label = "Solens absorptionslinjer")
                    break
                else:
                    ax.plot([i[1],i[1]],[offset,min_punkt],ls = ':',linewidth = 1.1, color = "gray")
                    break
                
    normalisered_sol = []
    x_axis = []
    for i in range(len(y)):
        if kalibreret_x[0]<=x[i]<=kalibreret_x[-1]:
            x_axis.append(x[i])
            if y[i]>1:
                normalisered_sol.append(y[i]*0.97-1)
            else: 
                normalisered_sol.append(y[i]-1)
    
    
    # clear_output()
    ax.plot(kalibreret_x,plaseret_skaleret_spektrum, linewidth = 1, color = "k")
    ax.plot(x_axis,normalisered_sol, linewidth = 1, color = "#1f77b4")
    # ax.legend(bbox_to_anchor=(0, 0.5), loc='center right')
    ax.legend()
    ax.set_xlabel("Bølgelængde (nm)")
    ax.set_xticks([400,425, 450,475, 500, 525, 550,575, 600,625, 650,675, 700])
    ax.set_yticklabels([])
    ax.set_yticks([])
    
    figur_mappe = "Grafer"
    figure_path = os.path.join(figur_mappe, f"Absorptionslinjer for {molekyle}.png")
    fig.savefig(figure_path)
    





    














# # Normaliser graf
# def normaliser(kalibreret_x, spektrum, afvigelse):
    
#     spline = fit(spektrum, afvigelse, korrekt = False)
    
#     offset = max(spektrum-spline)*0
#     fig, ax = plt.subplots(2,1,figsize = (12,6))
#     normalised_spectre = []
    
#     for i in range(0,len(spektrum)):
#         if spektrum[i]>spline[i]:
#             normalised_spectre.append((spektrum[i]-spline[i])**(2)*0.5+offset+0.5)
#         else :
#             normalised_spectre.append(-(spektrum[i]-spline[i])**(2)+offset+0.5)
    
#     procent_start = 0.10
#     procent_slut = 0.90
    
#     max_peak = np.max(np.abs(np.array(normalised_spectre[int(procent_start*len(normalised_spectre)):int(procent_slut*len(normalised_spectre))])-1-offset))
    
#     try:
#         start_index = np.where(np.abs(np.array(normalised_spectre[:int(procent_start*len(normalised_spectre))])-1-offset)>max_peak)[0][-1]+10
#         # print(start_index)
#     except: 
#         start_index = 0
    
#     try:
#         slut_index = np.where(np.abs(np.array(normalised_spectre[int(procent_slut*len(normalised_spectre)):])-1-offset)>max_peak)[0][0]+int(procent_slut*len(normalised_spectre))-10
#         # print(slut_index)
#     except:
#         slut_index = -1
    
#     start_index = 0
#     slut_index = -1
    
    
#     clear_output()
#     ax[0].plot(kalibreret_x[start_index:slut_index],spektrum[start_index:slut_index], linewidth = 1, color = "black", label = "Spektrum")
#     ax[0].plot(kalibreret_x[start_index:slut_index],spline[start_index:slut_index],linewidth = 2, color = "r", label = "Blød graf")
#     ax[0].set_title("Blød graf over Spektret")
#     ax[0].legend()
#     ax[0].set_xticks([])
#     ax[0].set_yticklabels([])
#     ax[0].set_yticks([])
    
#     ax[1].plot(kalibreret_x[start_index:slut_index], normalised_spectre[start_index:slut_index], color = "black",linewidth = 1)
#     ax[1].set_title("Normaliseret Spektrum (Spektrum/blød graf)")
#     ax[1].set_yticklabels([])
#     ax[1].set_yticks([])
#     ax[1].set_xlabel("Bølgelængde (nm)")
#     ax[1].set_xticks([400,425, 450,475, 500, 525, 550,575, 600,625, 650,675, 700])
    
#     return normalised_spectre
