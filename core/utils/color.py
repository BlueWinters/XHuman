

class Colors:
    """
    """
    Hex = (
        "FF3838",
        "2C99A8",
        "FF701F",
        "6473FF",
        "CFD231",
        "48F90A",
        "92CC17",
        "3DDB86",
        "1A9334",
        "00D4BB",
        "FF9D97",
        "00C2FF",
        "344593",
        "FFB21D",
        "0018EC",
        "8438FF",
        "520085",
        "CB38FF",
        "FF95C8",
        "FF37C7",
    )

    @staticmethod
    def getColor(index, bgr: bool = False):
        index = int(index) % len(Colors.Hex)
        color_codes = Colors.hex_to_rgb("#" + Colors.Hex[index])
        return (color_codes[2], color_codes[1], color_codes[0]) if bgr else color_codes

    @staticmethod
    def hex_to_rgb(hex_code):
        rgb = []
        for i in (0, 2, 4):
            rgb.append(int(hex_code[1 + i: 1 + i + 2], 16))
        return tuple(rgb)
