import cv2 as cv
import numpy as np
import random as rng
#from bright import increase_brightness

# Zmienna przechowująca kolejne numery obrazków.
img_number = [1,2,3,4,5,6,7,8,9,10,11,12] 

# Preprocessing zdjęcia
def preprocessing(img):
    # Zmiana wymiarów obrazka
    resized_img = rescaleFrame(img)
    cv.imshow('Zdjecie po zmianie wymiarow', resized_img)

    # Usuwanie efektu salt and pepper ze zdjęcia.
    median_blur= cv.medianBlur(resized_img, 7)

    # Zwiększenie jasności zdjęcia za pomocą funckji increase_brightness.
    bright = increase_brightness(median_blur, value = 80)
    
    # Wstępne przetworzenie obrazu oraz utworzenie najbardziej zewnętrznych kontórów zdjęcia obejmujących zewnętrzne krawędzie kart.
    gray = cv.cvtColor(bright, cv.COLOR_BGR2GRAY)
    threshold, thresh = cv.threshold(gray, 150, 255, cv.THRESH_OTSU)
    cv.imshow("Zdjecie z zewnetrznymi konturami kart", thresh)
    return thresh, resized_img, median_blur

# Funkcja do zmiany wymiarów zdjęcia.
def rescaleFrame(frame, scale_w=0.5, scale_h=0.5):
    width = int(frame.shape[1]*scale_w)
    height = int(frame.shape[0]*scale_h)
    dimensions = (width,height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

# Funkcja do obliczania momentów konturów.
def moment(contours):
    mu = [None]*len(contours)
    for i in range(len(contours)):
        mu[i] = cv.moments(contours[i])
    return mu

# Funkcja do obliczania środką masy konturów.
def mass_center(contours, mu):
    mc = [None]*len(contours)
    for i in range(len(contours)):
        # add 1e-5 to avoid division by zero
        mc[i] = (mu[i]['m10'] / (mu[i]['m00'] + 1e-5), mu[i]['m01'] / (mu[i]['m00'] + 1e-5))
    return mc

# Funkcja do rysowania konturów. 
def drawing_contours(contours,mc,canny):   
    drawing = np.zeros((canny.shape[0], canny.shape[1], 3), dtype=np.uint8)
    
    for i in range(len(contours)):
        color = (rng.randint(0,256), rng.randint(0,256), rng.randint(0,256))
        cv.drawContours(drawing, contours, i, color, 2)
        cv.circle(drawing, (int(mc[i][0]), int(mc[i][1])), 4, color, -1)
    return drawing

def increase_brightness(img, value=30):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv.merge((h, s, v))
    img = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
    return img

# Stworzenie maski
def mask(resized_img, contours, median_blur):
    blank = np.zeros(resized_img.shape[:2], dtype='uint8')
    for i in range(len(contours)):
        mask = cv.circle(blank, (int(mc[i][0]),int(mc[i][1])) ,85,255,-1)
        masked = cv.bitwise_and(median_blur,median_blur,mask=mask)
    return masked

# Klasyfikator rozjasniania maski.
# Klasyfikator umożliwa określenie wartości jasności danego zdjęcia i jedynie dla wartości granicznych zastosowania na nim funkcji zwiększającej jej jasność.
def brightness_classifier(resized_img, masked):
    hsv = cv.cvtColor(resized_img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    mean_brightness = hsv[...,2].mean()
    #print(mean_brightness)
    if mean_brightness < 75:
        masked_brighten = increase_brightness(masked, value = 30)
    else:
        masked_brighten = increase_brightness(masked, value = 0)
    return masked_brighten

# Obliczanie pół powierzchni konturów.
def contours_area(contours5):
    list = []
    for n in contours5:
        currentContour = n
        area = cv.contourArea(currentContour)
        list.append(area) 
    return list

# Filter konturów na podstawie hierarchii.
# n klasyfikuje rodzaj filtacji:
        # n == 0 : pierwsza iteracja
        # n == 1 : usunięcie elementów o zbyt małej powierzchni
        # n == 2 : wyznaczenie obrazu z zewnętrznymi konturami
def hierarchy_filter(hierarchies, contours,  resized_img, n, list = None):
    hierarchy = hierarchies[0] 
    i = 0
    blank = np.zeros(resized_img.shape[:2], dtype='uint8')
    for component in zip(contours, hierarchy):
        color = (rng.randint(150,256), rng.randint(0,256), rng.randint(0,256))
        currentContour = component[0]
        currentHierarchy = component[1]
        if n == 0:
            if currentHierarchy[2] < 20 and not currentHierarchy[3] < 0:
                cv.drawContours(blank, contours, i, color, 1)
            i = i+1
        elif n == 1:
            if list[i] > 1500:
                cv.drawContours(blank, contours, i, color, 1)
            i = i+1
        elif n == 2:
            if currentHierarchy[2] < 0:
                cv.drawContours(blank, contours, i, color, 1)
            i = i+1
    return blank

# Wyznaczenie ostatecznych zewnętrznych konturów kształtów.
def final_contours(resized_img, blank):
    mask = np.zeros(resized_img.shape[:2], dtype='uint8')
    blured_blank = cv.GaussianBlur(blank, (7,7), cv.BORDER_DEFAULT)
    threshold_blank, thresh_blank = cv.threshold(blured_blank, 150, 255, cv.THRESH_OTSU)
    contours, hierarchies = cv.findContours(thresh_blank, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours, hierarchies, mask

# Klasyfikator określający rodzaj symbolu.
# Klasyfikator określa rodzaj symbolu na podstawie momentów ostatecznych konturów.
def classifier(mu_blanked5, mc_blanked5):
    suma = 0
    special = 0 
    mc_reverse = 0
    mc_stop = 0
    for i in range(4):
        #print(f' MASS CENTER ({i}): {mc_blanked5[i]}')
        #print(mu_blanked5[i]['m00'])

        # Wartości graniczne dla symbolu '1'
        if mu_blanked5[i]['m00'] > 3000 and mu_blanked5[i]['m00'] < 4200:
            n = 1
            suma = suma + n
        elif mu_blanked5[i]['m00'] > 4200 and mu_blanked5[i]['m00'] < 5400:
            # Wartości graniczne dla symbolu '7'
            if mu_blanked5[i]['m10'] > 2000000 and mu_blanked5[i]['m10'] < 2700000:
                n = 7
                suma = suma + n 
            # Wartości graniczne dla symbolu 'Reverse'
            else:
                n = 0
                suma = suma + n 
                special = 1
                mc_reverse = mc_blanked5[i]
                #print(f'Special Card: Reverse // Mass Center {mc_blanked5[i]}')           
        elif mu_blanked5[i]['m00'] > 5400 and mu_blanked5[i]['m00'] < 7300: 
            # Wartości graniczne dla symbolu '5'
            if mu_blanked5[i]['m10'] > 1800000 and mu_blanked5[i]['m10'] < 2290000:
                n = 5
                suma = suma + n
            # Wartości graniczne dla symbolu '4'
            elif mu_blanked5[i]['m10'] > 2290000 and mu_blanked5[i]['m10'] < 2800000:
                n = 4
                suma = suma + n
            # Wartości graniczne dla symbolu '3'
            else:
                n = 3
                suma = suma + n 
        elif mu_blanked5[i]['m00'] > 7300 and mu_blanked5[i]['m00'] < 9000:
            # Wartości graniczne dla symbolu 'Stop'
            if mu_blanked5[i]['m10'] > 6700000 and mu_blanked5[i]['m10'] < 8000000:
                n = 0
                suma = suma + n
                special = 2
                mc_stop = mc_blanked5[i]
                #print(f'Special Card: Stop // Mass Center {mc_blanked5[i]}')
            # Wartości graniczne dla symbolu '8'
            else:
                n = 8 
                suma = suma + n
        #print(f' kolejna suma: {suma}')
    return suma, special, mc_reverse, mc_stop

# Określa rzeczywistą sume oraz rodzaj znaku
def real_values(img_n):
    if img_n > 0 and img_n <= 4:
        suma_n = 14
        special_n = 1 
    elif img_n > 4 and img_n <= 8:
        suma_n = 15
        special_n = 2
    else:
        suma_n = 14
        special_n = 0 
    return suma_n, special_n

# Pętle pozwalają na kolejne przetwarzanie obrazów.
# Dodatkowa pętla w zakresie 1 - 50 pozwala na wykonanie operacji na jednym obrazie 50 razy.
# Rozwiązanie zostało wprowadzone w celu upewnienia się, że kod będzie działał poprawnie za każdym razem, ponieważ wartości momentów mogą się różnić.

if __name__ == '__main__':
    for img_n in img_number:
        # Wgranie jednego zdjęcia
        img = cv.imread(f'images_path\\{img_n}.png')
        cv.imshow('Zdjęcie przed przetworzeniem', img)

        # Preprocessing zdjęcia
        thresh0, resized_img, median_blur = preprocessing(img)

        # Wyznaczenie konturów.
        contours, hierarchies = cv.findContours(thresh0, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Obliczenie momentów dla zewnętrznych konturów.
        mu = moment(contours)

        # Obliczenie środków kart na podstawie zewnętrznych konturów.
        mc = mass_center(contours, mu)

        # Stworzenie maski.
        # Maska ma postać okręgów o środkach pokrywających się ze środkiem karty.
        # Uzyskujemy obraz z widoczymi jedynie symbolami ze środków kart.
        masked = mask(resized_img, contours, median_blur)
        cv.imshow('Zdjecie z maska', masked)

        # Rozjaśnienie maski
        masked_brighten = brightness_classifier(resized_img, masked)

        #Operacje przetwarzające obraz z maską do konturowania oraz utworzenie konturów.
        gray2 = cv.cvtColor(masked_brighten, cv.COLOR_BGR2GRAY)
        cv.imshow('Zdjecie z maska z skali szarosci', gray2)
        blur2 = cv.GaussianBlur(gray2, (3,3), cv.BORDER_DEFAULT)
        cv.imshow('Zdjecie z maska z blurowaniem', blur2)
        threshold, thresh = cv.threshold(blur2, 150, 255, cv.THRESH_BINARY_INV)
        contours2, hierarchies2 = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

        # Wyznaczenie obrazu z uproszczonymi konturami - pierwsza iteracja.
        # Część konturów zostaje usunięta.
        # Algorytm jest wykonany na podstawie hierarchii.
        blank3 = hierarchy_filter(hierarchies2, contours2, resized_img, 0)
        cv.imshow('Zdjecie z usunieta cześcia konturow',blank3)

        # Ponowne utowrzenie konturów z obrobionego wyżej zdjęcia. 
        contours5, hierarchies5 = cv.findContours(blank3, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # Obliczenie pół powierzchni dla kontrurów
        list = contours_area(contours5)

        # Wyznaczenie obrazu z usunięciem wewnętrzych elementów o zbyt małym polu powierzchni.     
        blank4 = hierarchy_filter(hierarchies5, contours5, resized_img, 1, list = list)
        cv.imshow('Zdjecie z konturami z usunietymi wewnetrznymi elementami',blank4)

        # Obliczenie konturów.
        contours6, hierarchies6 = cv.findContours(blank4, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # Wyznaczenie obrazka z konturami zewnętrznymi. 
        blank5 = hierarchy_filter(hierarchies6, contours6, resized_img, 2)
        cv.imshow('Zdjecie z koncowymi konturami', blank5)

        # Wyznaczenie ostatecznych zewnętrznych konturów kształtów.
        contours7, hierarchies7, mask2 = final_contours(resized_img, blank5)

        # Obliczenie momentów oraz środków mas konturów.
        mu_blanked5 = moment(contours7)
        mc_blanked5 = mass_center(contours7,mu_blanked5)

        # Narysowanie ostatecznie obrobionego zdjęcia.
        drawing_blanked5 = drawing_contours(contours7, mc_blanked5, mask2 )
        cv.imshow('Ostateczne zdjecie po przetworzeniu', drawing_blanked5)

        # Klasyfikator określający rodzaj symbolu.
        suma, special, mc_reverse, mc_stop = classifier(mu_blanked5, mc_blanked5)

        # Określenie wartości rzeczywistej sumy oraz rodzaju znaku specjalnego.
        suma_n, special_n = real_values(img_n)

        # Sprawdzenie czy wartości obliczone przez klasyfikator zgadzają się z rzeczywistymi. 
        if suma == suma_n and special == special_n:
            pass
        else:
            for i in range(4):
                print(mu_blanked5[i]['m00'])
                print(mu_blanked5[i]['m10'])
            print(suma)

        cv.waitKey(0)
        
        # Drukowanie komunikatu o wyniku przetwarzania. 
        print(f'Dane dla zdjęcia {img_n}')
        print(f'Suma liczb na kartach: {suma}')
        if special == 1:
            print(f'Karta specjalna: Reverse')
            print(f'Środek karty: {mc_reverse}')
        elif special == 2:
            print(f'Karta specjalna: Stop')
            print(f'Środek karty: {mc_stop}')
        else:
            pass

print('Koniec')