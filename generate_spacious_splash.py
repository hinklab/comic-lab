import os
from PIL import Image, ImageDraw, ImageFont

def generate_spacious_splash():
    # Spacious dimensions: 780x460 (expanded from 640x380)
    w, h = 780, 460
    radius = 16

    # 1. Base canvas filled with magenta (Windows PyInstaller transparent color key)
    canvas = Image.new('RGB', (w, h), (255, 0, 255))

    # 2. Content surface
    content = Image.new('RGB', (w, h), (16, 17, 22))
    draw = ImageDraw.Draw(content)

    # Top accent line (3px) in vibrant coral (#ff3366)
    draw.rectangle([(0, 0), (w, 3)], fill=(255, 51, 102))

    # Bottom footer bar (70px tall, y = 390 to 460)
    footer_y = h - 70
    draw.rectangle([(0, footer_y), (w, h)], fill=(12, 13, 17))
    draw.line([(0, footer_y), (w, footer_y)], fill=(32, 36, 48), width=1)

    # Paste Logo Symbol at x=48, y=38
    logo_path = 'assets/comic-lab-logo.png'
    if os.path.exists(logo_path):
        logo = Image.open(logo_path).convert('RGBA')
        logo.thumbnail((92, 92), Image.Resampling.LANCZOS)
        content.paste(logo, (48, 38), logo)

    # Typography setup
    scale = 2
    font_size = 42
    fnt_bangers = ImageFont.truetype('fonts/Bangers-Regular.ttf', font_size * scale)
    fnt_inter_bold = ImageFont.truetype('fonts/Inter-VariableFont.ttf', 14)
    fnt_inter_reg = ImageFont.truetype('fonts/Inter-VariableFont.ttf', 13)
    fnt_inter_small = ImageFont.truetype('fonts/Inter-VariableFont.ttf', 12)

    # High-resolution rendering for COMIC-LAB and STUDIO
    title_canvas_w = 600 * scale
    title_canvas_h = 70 * scale
    t_img = Image.new('RGBA', (title_canvas_w, title_canvas_h), (0, 0, 0, 0))
    td = ImageDraw.Draw(t_img)

    tracking = int(1.2 * scale)
    sh_off = int(2.0 * scale)

    text1 = 'COMIC-LAB'
    text2 = 'STUDIO'
    gap_between = int(18 * scale)

    start_x = 24 * scale
    base_y = 12 * scale

    # Pass 1: Drop shadows for both words
    curr_x = start_x
    for ch in text1:
        ch_w = fnt_bangers.getbbox(ch)[2] - fnt_bangers.getbbox(ch)[0]
        td.text((curr_x + sh_off, base_y + sh_off), ch, font=fnt_bangers, fill=(0, 0, 0, 255))
        curr_x += ch_w + tracking

    curr_x += gap_between
    for ch in text2:
        ch_w = fnt_bangers.getbbox(ch)[2] - fnt_bangers.getbbox(ch)[0]
        td.text((curr_x + sh_off, base_y + sh_off), ch, font=fnt_bangers, fill=(0, 0, 0, 255))
        curr_x += ch_w + tracking

    # Pass 2: Main colored text: text1 in coral (#ff3366), text2 in pure white (#ffffff)
    curr_x = start_x
    for ch in text1:
        ch_w = fnt_bangers.getbbox(ch)[2] - fnt_bangers.getbbox(ch)[0]
        td.text((curr_x, base_y), ch, font=fnt_bangers, fill=(255, 51, 102, 255))
        curr_x += ch_w + tracking

    curr_x += gap_between
    for ch in text2:
        ch_w = fnt_bangers.getbbox(ch)[2] - fnt_bangers.getbbox(ch)[0]
        td.text((curr_x, base_y), ch, font=fnt_bangers, fill=(255, 255, 255, 255))
        curr_x += ch_w + tracking

    total_w = curr_x + 30 * scale
    cropped_t = t_img.crop((0, 0, total_w, title_canvas_h))

    # Downsample with Lanczos
    final_w = total_w // scale
    final_h = title_canvas_h // scale
    t_down = cropped_t.resize((final_w, final_h), Image.Resampling.LANCZOS)

    # Forward italic slant (shear = 0.24 matches CSS font-style: italic on web)
    pad_w = 40
    slanted_canvas = Image.new('RGBA', (final_w + pad_w, final_h), (0, 0, 0, 0))
    slanted_canvas.paste(t_down, (10, 0))
    slanted = slanted_canvas.transform(
        slanted_canvas.size,
        Image.AFFINE,
        (1, 0.24, 0, 0, 1, 0),
        resample=Image.Resampling.BICUBIC
    )

    # Paste title
    content.paste(slanted, (156, 36), slanted)

    # Subtitle
    draw.text((160, 94), 'Professional Comic Translation & Typesetting Suite', font=fnt_inter_reg, fill=(156, 163, 175))

    # Version info & status badge
    draw.text((160, 120), 'v1.0.0 (Windows 64-bit)', font=fnt_inter_small, fill=(107, 114, 128))
    draw.ellipse([(316, 126), (322, 132)], fill=(0, 230, 118))
    draw.text((330, 120), 'Offline Neural Engine Active', font=fnt_inter_small, fill=(0, 230, 118))

    # Horizontal divider between header and feature body
    draw.line([(48, 160), (w - 48, 160)], fill=(32, 36, 48), width=1)

    # 2 Columns of Features with ample spacing
    features_col1 = [
        ('100% Lossless HD Export', 'Asl sifatli va yuqori aniqlikdagi eksport'),
        ('Smart Bubble Detection', 'AI asosidagi kontur va pufakcha aniqlash'),
        ('60 FPS Accelerated Canvas', 'Tezkor, kechikishlarsiz interfaol tahrirlovchi')
    ]
    features_col2 = [
        ('Dynamic Comic Typography', 'Komikslar uchun moslashuvchan shriftlar'),
        ('Local In-Memory AI Cache', 'Takroriy tarjimalar uchun tezkor xotira'),
        ('Isolated Sandboxed Security', "Tizim xavfsizligi va ma'lumotlar himoyasi")
    ]

    col1_x = 48
    col2_x = 415
    start_y = 190
    item_gap = 56

    for i, (title, sub) in enumerate(features_col1):
        cy = start_y + i * item_gap
        # Emerald dot
        draw.ellipse([(col1_x, cy + 4), (col1_x + 8, cy + 12)], fill=(0, 230, 118))
        draw.text((col1_x + 22, cy), title, font=fnt_inter_bold, fill=(241, 245, 249))
        draw.text((col1_x + 22, cy + 20), sub, font=fnt_inter_small, fill=(100, 116, 139))

    for i, (title, sub) in enumerate(features_col2):
        cy = start_y + i * item_gap
        # Emerald dot
        draw.ellipse([(col2_x, cy + 4), (col2_x + 8, cy + 12)], fill=(0, 230, 118))
        draw.text((col2_x + 22, cy), title, font=fnt_inter_bold, fill=(241, 245, 249))
        draw.text((col2_x + 22, cy + 20), sub, font=fnt_inter_small, fill=(100, 116, 139))

    # Footer status indicator pulsing dot
    # Footer starts at footer_y = 390, ends at 460 (middle is 425)
    dot_center_y = footer_y + 35
    # Subtle outer ring
    draw.ellipse([(44, dot_center_y - 8), (60, dot_center_y + 8)], fill=(0, 230, 118, 50))
    # Core emerald dot
    draw.ellipse([(48, dot_center_y - 4), (56, dot_center_y + 4)], fill=(0, 230, 118))

    # Apply rounded corners mask without outer stroke
    mask = Image.new('L', (w, h), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle([(0, 0), (w, h)], radius=radius, fill=255)

    canvas.paste(content, (0, 0), mask)
    canvas.save('assets/splash.png')
    print('Successfully generated assets/splash.png (780x460)!')

if __name__ == '__main__':
    generate_spacious_splash()
