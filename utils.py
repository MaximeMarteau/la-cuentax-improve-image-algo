import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS']='lacuentax-adb671932cd7.json'
from google.cloud import vision

import numpy as np
import cv2
import matplotlib.pyplot as plt
import re
from unidecode import unidecode
import base64
import math

REG_PRICE = '(\+?)\d{1,}(\.|,)\d{2}\s' # white space at the end to exclude amounts with more than 2 decimals


def get_texts_coordinates_api(image_cv):
    """
    Consume google vision API to get image texts

    image_cv: cv2/numpy

    return: list of text with coordinates
    """
    client = vision.ImageAnnotatorClient(client_options={'api_endpoint': 'eu-vision.googleapis.com'})
    image_vision = vision.Image(content=cv2.imencode('.jpeg', image_cv)[1].tobytes())
        
    response = client.text_detection(image=image_vision)
    texts = response.text_annotations

    texts_coordinates = []

    for text in texts:
        vertices = (['({},{})'.format(vertex.x, vertex.y)
                    for vertex in text.bounding_poly.vertices])

        coordinates = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]

        texts_coordinates.append({
            'text': text.description,
            'coo': coordinates
        })

    return texts_coordinates


def align_text_boxes(texts_coordinates, display_image=False, image_cv=None):
    """
    Align text boxes on rows

    texts_coordinates: list of {text, coo}

    return: list of list of dict {text, coo, first_x, step}
    """
    # point 0: top left
    # point 1: top right
    # point 2: bottom right
    # point 3: bottom left
    
    # color cv2: (B, G, R)
    
    # Compute median height of boxes
    heights = []
    for text_coo in texts_coordinates:
        points = text_coo['coo']

        # Boxes are usually right so we can just measure the distance in the axis y
        heights.append(np.abs(points[1][1] - points[2][1]))
        
    median_height = np.median(heights)

    # Compute the y for each line
    # Create a new line when one box of text is too far from existing lines
    y_steps = []
    for text_coo in texts_coordinates:
        points = text_coo['coo']
        height_block = np.abs(points[1][1] - points[2][1])
        
        is_in_step = False
        for step in y_steps:
            
            # Gap between center of box and step < median height boxes
            if np.abs(points[0][1] + height_block / 2 - step) < median_height:
                text_coo['step'] = step
                is_in_step = True
                break

        if not is_in_step:
            y_steps.append(int(points[0][1] + height_block / 2))
            text_coo['step'] = int(points[0][1] + height_block / 2)

        text_coo['first_x'] = points[0][0]

    # Reorder boxes using the lines then its x
    texts_coordinates_sorted = sorted(texts_coordinates, key=lambda k: (k['step'], k['first_x']))
    
    final_texts_coordinates = []
    prev_step = 0
    new_row = []
    for text_coo in texts_coordinates_sorted:
        if text_coo['step'] != prev_step:
            prev_step = text_coo['step']
            if len(new_row):
                final_texts_coordinates.append(new_row)
            new_row = []
        new_row.append(text_coo)
    if len(new_row):
        final_texts_coordinates.append(new_row)

    # TODO: Check on each line if there are boxes with same x, ie vertically aligned
    # then move one boxe to the next/previous line

    
    # Draw text boxes and lines
    if display_image:
        image_draw = image_cv.copy()
            
        for text_coo in texts_coordinates:
            points = text_coo['coo']
            
            image_draw = cv2.line(image_draw, points[0], points[1], color=(255, 0, 0), thickness=10)
            image_draw = cv2.line(image_draw, points[1], points[2], color=(255, 0, 0), thickness=10)
            image_draw = cv2.line(image_draw, points[2], points[3], color=(255, 0, 0), thickness=10)
            image_draw = cv2.line(image_draw, points[3], points[0], color=(255, 0, 0), thickness=10)
            
        for y in y_steps:
            image_draw = cv2.line(image_draw, (0, y), (image_cv.shape[1], y), color=(0, 255, 0), thickness=10)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        axes[0].imshow(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
        axes[1].imshow(cv2.cvtColor(image_draw, cv2.COLOR_BGR2RGB))

    return final_texts_coordinates


def standardize_text(text):
    standard_text = unidecode(text)
    return standard_text.replace('.', '').upper()


def get_index_items_lines(text_lines, forbidden_words):
    first_true_added = False
    nb_following_lines_with_forbidden_words = 0
    lines_selected = []
    for text_line in text_lines:
        
        # If there are two following lines with forbidden words
        # We stop and don't look next lines
        if nb_following_lines_with_forbidden_words == 2:
            lines_selected.append(False)
            break
        
        contains_forbidden_words = False
        for forbidden_word in forbidden_words:
            if f' {forbidden_word}' in standardize_text(f' {text_line} ') or \
               f'{forbidden_word} ' in standardize_text(f' {text_line} '):
                contains_forbidden_words = True
                break
                
        if contains_forbidden_words and first_true_added: # Start this count only when we passed the first useless lines
            nb_following_lines_with_forbidden_words += 1
        else:
            nb_following_lines_with_forbidden_words = 0

        text_line_with_spaces = f' {text_line} '
        reg_letter = '[a-zA-Z]{2,}'
        
        if not contains_forbidden_words and \
           len([match.group() for match in re.finditer(REG_PRICE, text_line_with_spaces)]) > 0 and \
           len([match.group() for match in re.finditer(reg_letter, text_line_with_spaces)]) > 0:
            lines_selected.append(True)
            first_true_added = True
        else:
            lines_selected.append(False)
            
    start_index = lines_selected.index(max(lines_selected))
    end_index = lines_selected[::-1].index(max(lines_selected[::-1]))
    end_index = len(lines_selected) - end_index

    return start_index, end_index


def get_index_line_structure(text_lines):
    words_quantity = [
        'UNID',
        'UDS',
        'QTD', # pt
        'CANT'
    ]
    words_name = [
        'DESCRIPCION',
        'ARTICULO',
        'PRODUCTO',
        'ARTICLE', # fr
        'ARTIGO', # pt
    ]
    words_price_unity = [
        'PRECIO',
        'PVP',
        'PU', # fr
        'PRECO', # pt
    ]
    words_price_total = [
        'TOTAL',
        'IMPORTE',
        'VALOR', # pt
    ]
    
    # For each line check if words belong to one category
    for index, text in enumerate(text_lines):
        standard_text = standardize_text(text)
        is_in_each_cat = np.array([False, False, False, False])
        for word in standard_text.split():
            if word in words_quantity:
                is_in_each_cat[0] = True
            if word in words_name:
                is_in_each_cat[1] = True
            if word in words_price_unity:
                is_in_each_cat[2] = True
            if word in words_price_total:
                is_in_each_cat[3] = True
        
        # If one line contains words from at least 3 categories then it's the line
        if is_in_each_cat.sum() > 2:
            return index
                
    return -1


def get_index_items_lines_adjusted(text_lines):
    forbidden_words = [
        'TOTAL',
        'SUBTOTAL',
        'IVA',
        'IMP',
        'BASE',
        'CUOTA',
        'COBRO',
        'CONTADO',
        'TVA', # fr
        'COUVERT', # fr
        'MOYEN', # fr
        'INCL',
        'PAY',
        'NET',
        'SALES',
        'COMENSAL',
        'CAMARERO',
        'PAGO',
        'EFECTIVO',
        'CASH',
        'CARTE', # fr
        'ENTREGADO',
        'CAMBIO',
        'IUA', # OCR can read IUA instead of IVA
        'TUA', # OCR can read TUA instead of TVA
        # 'GETEILT', # ge, word used to show the amount divided by person
        # 'KARTENZAHLU', # ge, means paid by card
        'SALDO', # ge, total
        'NETTOUMSATZ', # ge, price without VAT
        'MWST', # ge, VAT
        'SERVICIOS',
        'ENTREGA',
        'SUMME', # ge, total
        'TRINKGELD', # ge, tips
        'KREDITKARTE', # ge, credit card
        'FACTURA'
    ]

    start_index, end_index = get_index_items_lines(text_lines, forbidden_words)
    index_line_structure = get_index_line_structure(text_lines)
    
    new_start = start_index
    new_end = end_index
    
    reg_letter = '[a-zA-Z]{2,}'
    
    if index_line_structure > -1:
        new_start = index_line_structure + 1
        
    # Check if we missed first amount, line with only amount and no letters
    elif len([match.group() for match in re.finditer(REG_PRICE, f' {text_lines[start_index - 1]}')]) > 0 and \
         len([match.group() for match in re.finditer(reg_letter, text_lines[start_index - 1])]) == 0:
        new_start -= 1

    for text_line in text_lines[end_index:]:
        contains_forbidden_words = False
        for forbidden_word in forbidden_words:
            if forbidden_word in standardize_text(text_line):
                contains_forbidden_words = True
                break
        if not contains_forbidden_words:
            new_end += 1
        else:
            break
    
    return new_start, new_end


def get_items(lines_items, verbose=False):
    items = []
    for line in lines_items:
        item = {
            'quantity': 1,
            'name': '???',
            'price': 0
        }
        
        line_with_spaces = f' {line} ' # In order to have two spaces around the quantity
        
        is_quantity_ok = False
        
        price_matches = [match.group() for match in re.finditer(REG_PRICE, line_with_spaces)]
        
        line_without_prices = line_with_spaces
        for price in price_matches:
            line_without_prices = line_without_prices.replace(price, '', 1)
        
        price_matches = [float(price.replace(',', '.')) for price in price_matches]
        
        if len(price_matches) == 0:
            if verbose:
                print('No price found')
                print(line)
        
        elif len(price_matches) == 1:
            item['price'] = price_matches[0]
            
        elif len(price_matches) == 2:
            item['price'] = price_matches[1]
        
            if price_matches[0] != 0 and \
               int(price_matches[1] * 100) % int(price_matches[0] * 100) == 0: # imprecision float
                
                item['quantity'] = int(price_matches[1] * 100) // int(price_matches[0] * 100)
                is_quantity_ok = True
            else:
                if verbose:
                    print('Two prices no divisibled', price_matches)
                    print(line)
        
        else:
            if verbose:
                print('Too many prices', price_matches)
                print(line)
        
        reg_quantity = '\s\d{1,}\s'
        quantity_matches = [match.group() for match in re.finditer(reg_quantity, line_with_spaces)]        
        quantity_matches_int = [int(quantity.replace(' ', '')) for quantity in quantity_matches \
                                if quantity.strip()[0] != '0']
        
        x_in_quantity = False

        if len(quantity_matches_int) == 0:
            reg_quantity_with_x = '\s(x\d{1,})|(\d{1,}x)\s'
            quantity_matches = [match.group() for match in re.finditer(reg_quantity_with_x, line_with_spaces)]
            quantity_matches_int = [int(quantity.replace(' ', '').replace('x', '')) for quantity in quantity_matches \
                                    if quantity.strip()[0] != '0']
            
            if len(quantity_matches_int) > 0:
              x_in_quantity = True
        
        if len(quantity_matches_int) == 1 and is_quantity_ok == True and \
           item['quantity'] != quantity_matches_int[0]:
            if verbose:
                print('Quantities don\'t match', item['quantity'], quantity_matches_int[0])
                print(line)
        
        if not is_quantity_ok:
            
            if len(quantity_matches_int) == 0:
                if verbose:
                    print('No quantity found')
                    print(line)
                
            elif len(quantity_matches_int) == 1:
                item['quantity'] = quantity_matches_int[0]
                
            else:
                if verbose:
                    print('Several quantities found, select min', quantity_matches_int)
                    print(line)
                item['quantity'] = min(quantity_matches_int)
        
        name = line_without_prices.replace('€', '')
        if not x_in_quantity:
            name = name.replace(f" {str(item['quantity'])} ", '').strip()
        else:
            if f" x{str(item['quantity'])} " in name:
                name = name.replace(f" x{str(item['quantity'])} ", '').strip()
            elif f" {str(item['quantity'])}x " in name:
                name = name.replace(f" {str(item['quantity'])}x ", '').strip()
        
        if name == '' and verbose:
            print('Name empty')
            print(line)
        else:
            item['name'] = name
            
        item['price'] = int(round(item['price'] * 100))        
        items.append(item)
    return items


def clean_text_boxes_no_price_aligned(text_boxes_aligned, nb_lines_to_delete, text_lines_selected):
    
    # Try to delete rows without quantities
    index_to_remove = []
    for index, row in enumerate(text_boxes_aligned):
        line = ' '.join([text_coo['text'] for text_coo in row])
        
        reg_quantity = '\s\d{1,}\s'
        if len([match.group() for match in re.finditer(reg_quantity, f' {line} ')]) == 0:
            index_to_remove.append(index)
    
    if len(index_to_remove) == nb_lines_to_delete:
        return [row for i, row in enumerate(text_boxes_aligned) if i not in index_to_remove]
    
    # Try to delete secondary rows (when contain C\.)
    index_to_remove = []
    for index, row in enumerate(text_boxes_aligned):
        line = ' '.join([text_coo['text'] for text_coo in row])
        if len([match.group() for match in re.finditer('C\.', f' {line} '.upper())]) > 0:
            index_to_remove.append(index)
            
    if len(index_to_remove) == nb_lines_to_delete:
        return [row for i, row in enumerate(text_boxes_aligned) if i not in index_to_remove]


    # Try to delete rows without prices using not realigned lines
    index_to_remove = []
    for index, line in enumerate(text_lines_selected):
        if len([match.group() for match in re.finditer(REG_PRICE, f' {line} ')]) == 0:
            index_to_remove.append(index)
    
    if len(index_to_remove) == nb_lines_to_delete:
        return [row for i, row in enumerate(text_boxes_aligned) if i not in index_to_remove]
        
        
    # Nothing perfectly matches so no change
    return text_boxes_aligned


def clean_items(items):
    print(len(items))
    # Remove IVA pourcentage in items' name
    reg_iva = '\d{2}%'
    items_without_iva = []
    for item in items:
        new_name = re.sub(reg_iva, '', item['name']).strip()
        items_without_iva.append({
            'quantity': item['quantity'],
            'name': new_name if new_name != '' else '???',
            'price': item['price']
        })
    
    # Remove word presents in all names (ex: 'x')
    common_words = items_without_iva[0]['name'].split()
    for item in items_without_iva[1:]:
        words = item['name'].split()
        common_words = list(set(words).intersection(common_words))
    
    if len(common_words) == 1:
        for i in range(len(items_without_iva)):

            # Add spaces around the special word to remove in order to not edit the correct name
            new_name = items_without_iva[i]['name'].replace(' ' + common_words[0], '').strip()
            new_name = new_name.replace(common_words[0] + ' ', '').strip()

            items_without_iva[i]['name'] = new_name if new_name != '' else '???'
    
    # Remove weird quantity in description (ex: 1,000)
    reg_weird_quantity = '\d{1,}(\.|,)000'
    items_without_weird_quantity = []
    for item in items_without_iva:
        new_name = re.sub(reg_weird_quantity, '', item['name']).strip()
        items_without_weird_quantity.append({
            'quantity': item['quantity'],
            'name': new_name if new_name != '' else '???',
            'price': item['price']
        })
        
    return items_without_weird_quantity


def evaluate_image_ocr(real_items, items_ocr):
    nb_items = len(real_items)
    nb_items_ocr = len(items_ocr)
    
    def compute_score(real_items, items_ocr):
        nb_correct_items = 0
        for real_item, item_ocr in zip(real_items, items_ocr):
            if real_item == item_ocr:
                nb_correct_items += 100
            else:
                if real_item['name'] == item_ocr['name']:
                    nb_correct_items += 30
                if real_item['quantity'] == item_ocr['quantity']:
                    nb_correct_items += 30
                if real_item['price'] == item_ocr['price']:
                    nb_correct_items += 30

        return nb_correct_items / (nb_items + np.abs(nb_items - nb_items_ocr))

    best_score = compute_score(real_items, items_ocr)
    
    # Compute score by shifting items
    if nb_items != nb_items_ocr:
        if nb_items > nb_items_ocr:
            score = compute_score(real_items[nb_items - nb_items_ocr:], items_ocr)
            
            if score > best_score:
                best_score = score
                
        elif nb_items < nb_items_ocr:
            score = compute_score(real_items, items_ocr[nb_items_ocr - nb_items:])
            
            if score > best_score:
                best_score = score

    return best_score


def full_ocr(image, texts_coordinates=None):

    # if not text_coordinates:
    #     texts_coordinates = get_texts_coordinates_api(image)
    #     print('get_texts_coordinates_api ended')

    text_boxes_aligned = align_text_boxes(texts_coordinates)
    print('align_text_boxes ended')

    text_lines = []
    for row in text_boxes_aligned:
        line = ''
        for text_coo in row:
            line += text_coo['text'] + ' '
        text_lines.append(line)

    start, end = get_index_items_lines_adjusted(text_lines)
    print('get_index_items_lines_adjusted ended')

    text_boxes_price = []
    text_boxes_no_price = []

    for row in text_boxes_aligned[start:end]:
        for text_coo in row:
            text_coo_without_eur = text_coo # .replace('OFERTA', '0,00')
            if '€' in text_coo['text']:
                text_coo_without_eur['text'] = text_coo['text'].replace('€', '')

            text_coo_without_eur['text'] = text_coo_without_eur['text'].replace('OFERTA', '0,00')
            
            # Here we add a whitespace at the beggining of the regex because
            # we are searching in text box (one word), not in text line
            if text_coo_without_eur['text'].strip() != '':            
                if len([match.group() for match in re.finditer('\s' + REG_PRICE, f" {text_coo_without_eur['text']} ")]) == 1:
                    text_boxes_price.append(text_coo_without_eur)
                else:
                    text_boxes_no_price.append(text_coo_without_eur)
      
    # Manage quantities with price format
    # Compute average abscissa of no price text boxes
    # If a text box price has a lower abscissa (on the left) then it's not a price
    xs = []
    for text_box in text_boxes_no_price:
        for points in text_box['coo']:
            xs.append(points[0])
    average_x_no_price = int(np.mean(xs))
    
    # Repeat this step bit with condition on average_x_no_price
    text_boxes_price = []
    text_boxes_no_price = []
    for row in text_boxes_aligned[start:end]:
        for text_coo in row:
            text_coo_without_eur = text_coo # .replace('OFERTA', '0,00')
            if '€' in text_coo['text']:
                text_coo_without_eur['text'] = text_coo['text'].replace('€', '')

            text_coo_without_eur['text'] = text_coo_without_eur['text'].replace('OFERTA', '0,00')
            
            if text_coo_without_eur['text'].strip() != '':            
                if len([match.group() for match in re.finditer('\s' + REG_PRICE, f" {text_coo_without_eur['text']} ")]) == 1:
                    if text_coo_without_eur['first_x'] > average_x_no_price:
                        text_boxes_price.append(text_coo_without_eur)
                    else:
                        text_coo_without_eur['text'] = re.split(r'(\.|,)', text_coo_without_eur['text'])[0]
                        text_boxes_no_price.append(text_coo_without_eur)
                else:
                    text_boxes_no_price.append(text_coo_without_eur)


    # For prices realign alone
    text_boxes_price_aligned = align_text_boxes(text_boxes_price, False, image)

    # Remove prices to get names and quantities
    text_boxes_no_price_aligned = []
    steps_row_containing_price = [] # To manage secondary lines (pt)
    for row in text_boxes_aligned[start:end]:
        new_row = []
        was_containing_price = False # To manage secondary lines
        for text_coo in row:
            if text_coo in text_boxes_price: # To manage secondary lines
                was_containing_price = True
            if text_coo in text_boxes_no_price and text_coo['text'] != '':
                new_row.append(text_coo)
        if was_containing_price: # To manage secondary lines
            steps_row_containing_price.append(row[0]['step'])
        if len(new_row):
            text_boxes_no_price_aligned.append(new_row)

    print('Text boxes with and without prices separated')

    # When there are too many secondary lines, keep only lines matching prices
    if len(text_boxes_no_price_aligned) > 1.5 * len(text_boxes_price_aligned):                
        new_text_boxes = []
        for row in text_boxes_no_price_aligned:
            if row[0]['step'] in steps_row_containing_price:
                new_text_boxes.append(row)
        text_boxes_no_price_aligned = new_text_boxes

    if len(text_boxes_no_price_aligned) > len(text_boxes_price_aligned):
        text_boxes_no_price_aligned = clean_text_boxes_no_price_aligned(text_boxes_no_price_aligned,
                                        len(text_boxes_no_price_aligned) - len(text_boxes_price_aligned),
                                        text_lines[start:end])

    text_lines = []
    for row_no_price, row_price in zip(text_boxes_no_price_aligned, text_boxes_price_aligned):
        line = ' '.join([text_coo['text'] for text_coo in row_no_price] + \
                        [text_coo['text'] for text_coo in row_price])
        text_lines.append(line)

    print('Remove secondary lines')
        
    items_ocr = get_items(text_lines)
    print('get_items ended')

    print(items_ocr)

    items_ocr = clean_items(items_ocr)
    print('clean_items ended')

    return items_ocr

def full_ocr_v2(image, texts_coordinates=None):
    image_height = image.shape[0]
    image_width = image.shape[1]
    texts_coordinates, std_angle = rotate_text_coordinates(texts_coordinates, image_height, image_width)
    res = full_ocr(image, texts_coordinates)
    return res, std_angle


def rotate_point(x, y, center_x, center_y, angle_radians):
    # Translate point to origin relative to center
    translated_x = x - center_x
    translated_y = y - center_y
    
    # Rotate (clockwise correction, so negate angle)
    new_x = translated_x * math.cos(-angle_radians) - translated_y * math.sin(-angle_radians)
    new_y = translated_x * math.sin(-angle_radians) + translated_y * math.cos(-angle_radians)
    
    # Translate back
    return round(new_x + center_x), round(new_y + center_y)

def calculate_rotation_angle(box):
    # Compute angle using top edge 
    dx = box[1][0] - box[0][0]
    dy = box[1][1] - box[0][1]
    return math.atan2(dy, dx)

def rotate_text_coordinates(texts_coordinates, image_height, image_width):
    # Collect angles from all boxes
    angles = []
    for item in texts_coordinates:
        box = item['coo']
        angle = calculate_rotation_angle(box)
        angles.append(angle)

    # Compute the average angle
    average_angle = sum(angles) / len(angles)

    std_angle = np.std(angles)

    print('Too high std in angles, aborte rotating')
    if std_angle > .1:
        return texts_coordinates, std_angle

    # Define the rotation center (image center)
    center_x, center_y = image_width / 2, image_height / 2

    # Collect original and rotated bounding boxes and words for line detection
    # original_boxes = []
    rotated_texts_coordinates = []
    for item in texts_coordinates:
        box = item['coo']
        rotated_box = [
            rotate_point(point[0], point[1], center_x, center_y, average_angle)
            for point in box
        ]
        rotated_texts_coordinates.append({
            'coo': rotated_box,
            'text': item['text']
        })
    return rotated_texts_coordinates, std_angle
