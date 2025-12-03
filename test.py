import qrcode

# URL to encode
url = "https://github.com/xavierhillroy/LGP_Vision"

# Create QR code instance
qr = qrcode.QRCode(
    version=1,  # Controls the size of the QR Code (1 is the smallest)
    error_correction=qrcode.constants.ERROR_CORRECT_L,  # Error correction level
    box_size=10,  # Size of each box in pixels
    border=4,  # Border thickness in boxes
)

# Add data to QR code
qr.add_data(url)
qr.make(fit=True)

# Create image from QR code
img = qr.make_image(fill_color="black", back_color="white")

# Save as PNG
img.save("qrcode.png")
print(f"✓ QR code generated and saved as 'qrcode.png' for: {url}")

