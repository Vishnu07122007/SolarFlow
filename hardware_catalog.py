"""
SolarFlow - Hardware Catalog
============================

Clean, standalone hardware catalog for the new SolarFlow application.

Design goals
------------
1. Keep the supplied Indian panel/inverter catalogue.
2. Optionally extend it with pvlib's CEC databases when available.
3. Never silently invent a panel/inverter when an unknown model is requested.
4. Keep nameplate power, temperature coefficient and efficiency separate.
5. Expose explicit bifacial metadata for the supplied bifacial products.
6. Provide a stable API for the Flask app and forecast engine.

Important:
- Panel STC_W is nameplate electrical power at STC.
- gamma_r is the power temperature coefficient in fraction / °C.
  Example: -0.30 means -0.30 %/°C = -0.0030 /°C.
- Efficiency is stored as percentage metadata and is NOT multiplied into
  nameplate power again.
- Inverter efficiency returned here is a nominal DC->AC efficiency derived
  from Paco/Pdco. It is not a manufacturer efficiency curve.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Optional, Tuple
import math
import re

import pandas as pd



# ---------------------------------------------------------------------------
# Supplied Indian/custom catalogue
# ---------------------------------------------------------------------------

INDIAN_CUSTOM_PANELS = pd.DataFrame(
    [
    # -------------------- Waaree --------------------
    {"Name": "🇮🇳 Waaree 400W Mono PERC", "STC_W": 400.0, "gamma_r": -0.35, "Efficiency": 20.4, },
    {"Name": "🇮🇳 Waaree 440W Mono PERC", "STC_W": 440.0, "gamma_r": -0.34, "Efficiency": 20.8, },
    {"Name": "🇮🇳 Waaree 450W Mono PERC", "STC_W": 450.0, "gamma_r": -0.34, "Efficiency": 20.9, },
    {"Name": "🇮🇳 Waaree 530W Mono PERC", "STC_W": 530.0, "gamma_r": -0.34, "Efficiency": 21.0, },
    {"Name": "🇮🇳 Waaree 540W TOPCon Bifacial", "STC_W": 540.0, "gamma_r": -0.30, "Efficiency": 21.6, },
    {"Name": "🇮🇳 Waaree 540W TOPCon", "STC_W": 540.0, "gamma_r": -0.30, "Efficiency": 21.6, },
    {"Name": "🇮🇳 Waaree 550W TOPCon", "STC_W": 550.0, "gamma_r": -0.30, "Efficiency": 21.8, },
    {"Name": "🇮🇳 Waaree 565W Elite BiN", "STC_W": 565.0, "gamma_r": -0.30, "Efficiency": 21.9, },
    {"Name": "🇮🇳 Waaree Elite BiN 565W", "STC_W": 565.0, "gamma_r": -0.30, "Efficiency": 21.9, },
    {"Name": "🇮🇳 Waaree 580W N-Type TOPCon", "STC_W": 580.0, "gamma_r": -0.30, "Efficiency": 22.5, },
    {"Name": "🇮🇳 Waaree 580W TOPCon", "STC_W": 580.0, "gamma_r": -0.30, "Efficiency": 22.5, },
    {"Name": "🇮🇳 Waaree 590W Elite", "STC_W": 590.0, "gamma_r": -0.30, "Efficiency": 22.8, },
    {"Name": "🇮🇳 Waaree 600W Elite BiN", "STC_W": 600.0, "gamma_r": -0.30, "Efficiency": 23.2, },
    {"Name": "🇮🇳 Waaree Elite BiN 600W", "STC_W": 600.0, "gamma_r": -0.30, "Efficiency": 23.2, },
    {"Name": "🇮🇳 Waaree 620W Bifacial", "STC_W": 620.0, "gamma_r": -0.29, "Efficiency": 22.8, },

    # -------------------- Adani --------------------
    {"Name": "🇮🇳 Adani 400W Shine PERC", "STC_W": 400.0, "gamma_r": -0.35, "Efficiency": 20.2, },
    {"Name": "🇮🇳 Adani 440W Shine PERC", "STC_W": 440.0, "gamma_r": -0.35, "Efficiency": 20.8, },
    {"Name": "🇮🇳 Adani 450W Shine", "STC_W": 450.0, "gamma_r": -0.35, "Efficiency": 21.0, },
    {"Name": "🇮🇳 Adani 530W Bifacial", "STC_W": 530.0, "gamma_r": -0.32, "Efficiency": 21.0, },
    {"Name": "🇮🇳 Adani 540W Bifacial", "STC_W": 540.0, "gamma_r": -0.32, "Efficiency": 21.2, },
    {"Name": "🇮🇳 Adani 550W TOPCon", "STC_W": 550.0, "gamma_r": -0.30, "Efficiency": 21.4, },
    {"Name": "🇮🇳 Adani 570W Elan TOPCon", "STC_W": 570.0, "gamma_r": -0.30, "Efficiency": 21.5, },
    {"Name": "🇮🇳 Adani Elan 570W", "STC_W": 570.0, "gamma_r": -0.30, "Efficiency": 21.5, },
    {"Name": "🇮🇳 Adani 580W TOPCon", "STC_W": 580.0, "gamma_r": -0.30, "Efficiency": 22.3, },
    {"Name": "🇮🇳 Adani 590W TOPCon", "STC_W": 590.0, "gamma_r": -0.30, "Efficiency": 22.4, },

    # -------------------- Tata Power Solar --------------------
    {"Name": "🇮🇳 Tata Power 400W Mono", "STC_W": 400.0, "gamma_r": -0.36, "Efficiency": 20.3, },
    {"Name": "🇮🇳 Tata Power 440W Mono", "STC_W": 440.0, "gamma_r": -0.36, "Efficiency": 20.5, },
    {"Name": "🇮🇳 Tata Power 530W Mono PERC", "STC_W": 530.0, "gamma_r": -0.36, "Efficiency": 20.5, },
    {"Name": "🇮🇳 Tata Power 540W Mono", "STC_W": 540.0, "gamma_r": -0.35, "Efficiency": 21.0, },
    {"Name": "🇮🇳 Tata Power 550W TOPCon", "STC_W": 550.0, "gamma_r": -0.32, "Efficiency": 21.3, },
    {"Name": "🇮🇳 Tata Power 580W TOPCon", "STC_W": 580.0, "gamma_r": -0.30, "Efficiency": 22.1, },

    # -------------------- Vikram Solar --------------------
    {"Name": "🇮🇳 Vikram 440W Eldora", "STC_W": 440.0, "gamma_r": -0.34, "Efficiency": 20.8, },
    {"Name": "🇮🇳 Vikram 540W Ultima PERC", "STC_W": 540.0, "gamma_r": -0.34, "Efficiency": 21.1, },
    {"Name": "🇮🇳 Vikram Ultima 540W", "STC_W": 540.0, "gamma_r": -0.34, "Efficiency": 21.1, },
    {"Name": "🇮🇳 Vikram 550W Hypersol", "STC_W": 550.0, "gamma_r": -0.30, "Efficiency": 21.5, },
    {"Name": "🇮🇳 Vikram 580W Hypersol TOPCon", "STC_W": 580.0, "gamma_r": -0.30, "Efficiency": 22.5, },
    {"Name": "🇮🇳 Vikram Hypersol 580W", "STC_W": 580.0, "gamma_r": -0.30, "Efficiency": 22.5, },
    {"Name": "🇮🇳 Vikram 590W Hypersol", "STC_W": 590.0, "gamma_r": -0.30, "Efficiency": 22.6, },
    {"Name": "🇮🇳 Vikram 700W Hypersol G12", "STC_W": 700.0, "gamma_r": -0.30, "Efficiency": 22.5, },
    {"Name": "🇮🇳 Vikram 715W Hypersol G12", "STC_W": 715.0, "gamma_r": -0.30, "Efficiency": 23.0, },

    # -------------------- Premier Energies --------------------
    {"Name": "🇮🇳 Premier 540W Mono PERC", "STC_W": 540.0, "gamma_r": -0.35, "Efficiency": 21.0, },
    {"Name": "🇮🇳 Premier 550W Mono PERC", "STC_W": 550.0, "gamma_r": -0.35, "Efficiency": 21.3, },
    {"Name": "🇮🇳 Premier 600W TOPCon", "STC_W": 600.0, "gamma_r": -0.29, "Efficiency": 22.2, },
    {"Name": "🇮🇳 Premier 620W TOPCon NeoBlack", "STC_W": 620.0, "gamma_r": -0.29, "Efficiency": 22.95, },
    {"Name": "🇮🇳 Premier NeoBlack 620W", "STC_W": 620.0, "gamma_r": -0.29, "Efficiency": 22.95, },
    {"Name": "🇮🇳 Premier 630W TOPCon", "STC_W": 630.0, "gamma_r": -0.29, "Efficiency": 23.3, },

    # -------------------- Goldi Solar --------------------
    {"Name": "🇮🇳 Goldi 530W Mono", "STC_W": 530.0, "gamma_r": -0.35, "Efficiency": 20.5, },
    {"Name": "🇮🇳 Goldi 540W Mono PERC", "STC_W": 540.0, "gamma_r": -0.34, "Efficiency": 20.9, },
    {"Name": "🇮🇳 Goldi 550W Mono", "STC_W": 550.0, "gamma_r": -0.34, "Efficiency": 21.2, },
    {"Name": "🇮🇳 Goldi 665W IBC", "STC_W": 665.0, "gamma_r": -0.26, "Efficiency": 24.6, },

    # -------------------- RenewSys --------------------
    {"Name": "🇮🇳 RenewSys 540W Mono", "STC_W": 540.0, "gamma_r": -0.34, "Efficiency": 20.8, },
    {"Name": "🇮🇳 RenewSys 550W Mono", "STC_W": 550.0, "gamma_r": -0.34, "Efficiency": 21.1, },
    {"Name": "🇮🇳 RenewSys 580W TOPCon", "STC_W": 580.0, "gamma_r": -0.30, "Efficiency": 22.0, },

    # -------------------- Emmvee --------------------
    {"Name": "🇮🇳 Emmvee 535W Bifacial", "STC_W": 535.0, "gamma_r": -0.35, "Efficiency": 20.7, },
    {"Name": "🇮🇳 Emmvee 540W Bifacial", "STC_W": 540.0, "gamma_r": -0.35, "Efficiency": 20.9, },
    {"Name": "🇮🇳 Emmvee 550W Bifacial", "STC_W": 550.0, "gamma_r": -0.35, "Efficiency": 21.3, },

    # -------------------- Luminous --------------------
    {"Name": "🇮🇳 Luminous 330W Poly", "STC_W": 330.0, "gamma_r": -0.38, "Efficiency": 16.9, },
    {"Name": "🇮🇳 Luminous 395W Mono PERC", "STC_W": 395.0, "gamma_r": -0.35, "Efficiency": 19.9, },
    {"Name": "🇮🇳 Luminous 540W Mono PERC", "STC_W": 540.0, "gamma_r": -0.35, "Efficiency": 20.9, },
    {"Name": "🇮🇳 Luminous 540W", "STC_W": 540.0, "gamma_r": -0.35, "Efficiency": 20.9, },
    {"Name": "🇮🇳 Luminous 550W Mono PERC", "STC_W": 550.0, "gamma_r": -0.35, "Efficiency": 21.0, },
    {"Name": "🇮🇳 Luminous 565W TOPCon", "STC_W": 565.0, "gamma_r": -0.31, "Efficiency": 21.9, },
    {"Name": "🇮🇳 Luminous 620W TOPCon", "STC_W": 620.0, "gamma_r": -0.31, "Efficiency": 22.95, },
    {"Name": "🇮🇳 Luminous 630W TOPCon", "STC_W": 630.0, "gamma_r": -0.31, "Efficiency": 23.3, },
    {"Name": "🇮🇳 Luminous 635W TOPCon", "STC_W": 635.0, "gamma_r": -0.31, "Efficiency": 23.5, },
    {"Name": "🇮🇳 Luminous 640W TOPCon", "STC_W": 640.0, "gamma_r": -0.29, "Efficiency": 22.9, },
    {"Name": "🇮🇳 Luminous 700W TOPCon", "STC_W": 700.0, "gamma_r": -0.29, "Efficiency": 22.5, },

    # -------------------- Other Indian --------------------
    {"Name": "🇮🇳 Loom Solar 540W Shark", "STC_W": 540.0, "gamma_r": -0.34, "Efficiency": 21.5, },
    {"Name": "🇮🇳 Loom Solar 540W", "STC_W": 540.0, "gamma_r": -0.34, "Efficiency": 21.5, },
    {"Name": "🇮🇳 Navitas 540W Mono", "STC_W": 540.0, "gamma_r": -0.35, "Efficiency": 21.0, },
    {"Name": "🇮🇳 Jakson 550W Helia", "STC_W": 550.0, "gamma_r": -0.34, "Efficiency": 21.2, },
    {"Name": "🇮🇳 Jakson 580W Mono", "STC_W": 580.0, "gamma_r": -0.34, "Efficiency": 21.5, },

    # -------------------- Global Brands --------------------
    {"Name": "Jinko Tiger Neo 540W", "STC_W": 540.0, "gamma_r": -0.29, "Efficiency": 21.5, },
    {"Name": "Jinko Tiger Neo 580W", "STC_W": 580.0, "gamma_r": -0.29, "Efficiency": 22.5, },
    {"Name": "Jinko Tiger Neo 590W", "STC_W": 590.0, "gamma_r": -0.29, "Efficiency": 22.8, },
    {"Name": "Jinko Tiger Neo 600W", "STC_W": 600.0, "gamma_r": -0.29, "Efficiency": 23.0, },
    {"Name": "Jinko 580W Tiger Neo", "STC_W": 580.0, "gamma_r": -0.29, "Efficiency": 22.5, },
    {"Name": "LONGi Hi-MO6 550W", "STC_W": 550.0, "gamma_r": -0.29, "Efficiency": 21.5, },
    {"Name": "LONGi Hi-MO6 580W", "STC_W": 580.0, "gamma_r": -0.29, "Efficiency": 22.5, },
    {"Name": "LONGi 580W Hi-MO6", "STC_W": 580.0, "gamma_r": -0.29, "Efficiency": 22.5, },
    {"Name": "Canadian Solar 550W", "STC_W": 550.0, "gamma_r": -0.30, "Efficiency": 21.5, },
    {"Name": "Trina Vertex 550W", "STC_W": 550.0, "gamma_r": -0.30, "Efficiency": 21.4, },
    {"Name": "JA Solar 540W", "STC_W": 540.0, "gamma_r": -0.30, "Efficiency": 21.2, },
]).set_index("Name")


INDIAN_CUSTOM_INVERTERS = pd.DataFrame(
   [
    # -------------------- Luminous --------------------
    {"Name": "🇮🇳 Luminous NXi 1kW", "Paco": 1000.0, "Pdco": 1200.0},
    {"Name": "🇮🇳 Luminous NXi 2kW", "Paco": 2000.0, "Pdco": 2300.0},
    {"Name": "🇮🇳 Luminous NXi 3kW", "Paco": 3000.0, "Pdco": 3500.0},
    {"Name": "🇮🇳 Luminous NXi 4kW", "Paco": 4000.0, "Pdco": 4600.0},
    {"Name": "🇮🇳 Luminous NXi 5kW", "Paco": 5000.0, "Pdco": 5800.0},
    {"Name": "🇮🇳 Luminous NXi-5kW", "Paco": 5000.0, "Pdco": 5800.0},
    {"Name": "🇮🇳 Luminous NXI 5000", "Paco": 5000.0, "Pdco": 5800.0},
    {"Name": "🇮🇳 Luminous Grid Tie 3kW", "Paco": 3000.0, "Pdco": 3100.0},
    {"Name": "🇮🇳 Luminous Grid Tie 5kW", "Paco": 5000.0, "Pdco": 5150.0},
    {"Name": "🇮🇳 Luminous Hybrid 5kW", "Paco": 5000.0, "Pdco": 5200.0},

    # -------------------- Power-One --------------------
    {"Name": "🇮🇳 Power-One SGTU-101N 1kW", "Paco": 1000.0, "Pdco": 1100.0},
    {"Name": "🇮🇳 Power-One SGTU-102N 2kW", "Paco": 2000.0, "Pdco": 2200.0},
    {"Name": "🇮🇳 Power-One SGTU-103N 3kW", "Paco": 3000.0, "Pdco": 3300.0},
    {"Name": "🇮🇳 Power-One SGTU-105N 5kW", "Paco": 5000.0, "Pdco": 5500.0},
    {"Name": "🇮🇳 Power-One SGTU-105N", "Paco": 5000.0, "Pdco": 5500.0},
    {"Name": "🇮🇳 Power-One SGTU-106N 6kW", "Paco": 6000.0, "Pdco": 6600.0},
    {"Name": "🇮🇳 Power-One SGTU-106N", "Paco": 6000.0, "Pdco": 6600.0},

    # -------------------- V-Guard --------------------
    {"Name": "🇮🇳 V-Guard SolSmart 3000 3kW", "Paco": 3000.0, "Pdco": 3100.0},
    {"Name": "🇮🇳 V-Guard SolSmart 5000 5kW", "Paco": 5000.0, "Pdco": 5200.0},
    {"Name": "🇮🇳 V-Guard SolSmart 5000", "Paco": 5000.0, "Pdco": 5200.0},
    {"Name": "🇮🇳 V-Guard SolSmart 5000 GFI", "Paco": 5000.0, "Pdco": 5200.0},
    {"Name": "🇮🇳 V-Guard SolSmart 6000T 6kW", "Paco": 6000.0, "Pdco": 6200.0},
    {"Name": "🇮🇳 V-Guard Solrigo 5000 5kW", "Paco": 5000.0, "Pdco": 5150.0},
    {"Name": "🇮🇳 V-Guard Solrigo 5000", "Paco": 5000.0, "Pdco": 5150.0},

    # -------------------- Okaya --------------------
    {"Name": "🇮🇳 Okaya GTI 3kW", "Paco": 3000.0, "Pdco": 3600.0},
    {"Name": "🇮🇳 Okaya GTI 3.4kW", "Paco": 3400.0, "Pdco": 4080.0},
    {"Name": "🇮🇳 Okaya GTI 5kW", "Paco": 5000.0, "Pdco": 6000.0},
    {"Name": "🇮🇳 Okaya GTI 5KW", "Paco": 5000.0, "Pdco": 6000.0},
    {"Name": "🇮🇳 Okaya SOLAR GTI 5KW P1", "Paco": 5000.0, "Pdco": 6000.0},
    {"Name": "🇮🇳 Okaya GTI 5kW 3-Phase", "Paco": 5000.0, "Pdco": 6000.0},
    {"Name": "🇮🇳 Okaya GTI 15kW 3-Phase", "Paco": 15000.0, "Pdco": 18000.0},

    # -------------------- Other Indian --------------------
    {"Name": "🇮🇳 Havells Enviro 3kW", "Paco": 3000.0, "Pdco": 3100.0},
    {"Name": "🇮🇳 Havells Enviro 5kW", "Paco": 5000.0, "Pdco": 5150.0},
    {"Name": "🇮🇳 Microtek 3kW", "Paco": 3000.0, "Pdco": 3100.0},
    {"Name": "🇮🇳 Microtek 5kW", "Paco": 5000.0, "Pdco": 5200.0},
    {"Name": "🇮🇳 Polycab 5kW", "Paco": 5000.0, "Pdco": 5150.0},

    # -------------------- Growatt --------------------
    {"Name": "Growatt MIN 3kW", "Paco": 3000.0, "Pdco": 3060.0},
    {"Name": "Growatt MIN 5000TL-X", "Paco": 5000.0, "Pdco": 5100.0},
    {"Name": "Growatt MIN 5000TL-XH", "Paco": 5000.0, "Pdco": 5100.0},
    {"Name": "Growatt MIN 5kW", "Paco": 5000.0, "Pdco": 5100.0},
    {"Name": "Growatt MIN 6kW", "Paco": 6000.0, "Pdco": 6120.0},
    {"Name": "Growatt MIN 6000TL-X", "Paco": 6000.0, "Pdco": 6120.0},
    {"Name": "Growatt MIN 8kW", "Paco": 8000.0, "Pdco": 8160.0},
    {"Name": "Growatt MIN 8000TL-X", "Paco": 8000.0, "Pdco": 8160.0},

    # -------------------- Sungrow --------------------
    {"Name": "Sungrow SG 3kW", "Paco": 3000.0, "Pdco": 3060.0},
    {"Name": "Sungrow SG5.0RS", "Paco": 5000.0, "Pdco": 5100.0},
    {"Name": "Sungrow SG5RS", "Paco": 5000.0, "Pdco": 5100.0},
    {"Name": "Sungrow SG 5kW", "Paco": 5000.0, "Pdco": 5100.0},
    {"Name": "Sungrow SG6.0RS", "Paco": 6000.0, "Pdco": 6120.0},
    {"Name": "Sungrow SG 6kW", "Paco": 6000.0, "Pdco": 6120.0},
    {"Name": "Sungrow SG10RS", "Paco": 10000.0, "Pdco": 10200.0},
    {"Name": "Sungrow SG 10kW", "Paco": 10000.0, "Pdco": 10200.0},

    # -------------------- Solis --------------------
    {"Name": "Solis 5kW", "Paco": 5000.0, "Pdco": 5150.0},
    {"Name": "Solis S6-GR1P5K", "Paco": 5000.0, "Pdco": 5150.0},
    {"Name": "Solis-5K", "Paco": 5000.0, "Pdco": 5150.0},
    {"Name": "Solis 6kW", "Paco": 6000.0, "Pdco": 6180.0},
    {"Name": "Solis S6-GR1P6K", "Paco": 6000.0, "Pdco": 6180.0},

    # -------------------- GoodWe --------------------
    {"Name": "GoodWe 5kW", "Paco": 5000.0, "Pdco": 5120.0},
    {"Name": "GoodWe GW5000-MS", "Paco": 5000.0, "Pdco": 5120.0},
    {"Name": "GoodWe 6kW", "Paco": 6000.0, "Pdco": 6150.0},
    {"Name": "GoodWe GW6000-MS", "Paco": 6000.0, "Pdco": 6150.0},

    # -------------------- Deye --------------------
    {"Name": "Deye 5kW", "Paco": 5000.0, "Pdco": 5120.0},
    {"Name": "Deye SUN-5K-G", "Paco": 5000.0, "Pdco": 5120.0},
    {"Name": "Deye 6kW", "Paco": 6000.0, "Pdco": 6150.0},
    {"Name": "Deye SUN-6K-G", "Paco": 6000.0, "Pdco": 6150.0},

    # -------------------- Fronius --------------------
    {"Name": "Fronius Primo 5kW", "Paco": 5000.0, "Pdco": 5100.0},
    {"Name": "Fronius Primo 5.0-1", "Paco": 5000.0, "Pdco": 5100.0},
    {"Name": "Fronius Symo 5kW", "Paco": 5000.0, "Pdco": 5100.0},
    {"Name": "Fronius Symo 5.0-3", "Paco": 5000.0, "Pdco": 5100.0},

    # -------------------- Others --------------------
    {"Name": "Delta 5kW", "Paco": 5000.0, "Pdco": 5120.0},
    {"Name": "Huawei 5kW", "Paco": 5000.0, "Pdco": 5100.0},
    {"Name": "Huawei SUN2000-5KTL", "Paco": 5000.0, "Pdco": 5100.0},
]).set_index("Name")


# ---------------------------------------------------------------------------
# Normalized public records
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PanelSpec:
    name: str
    stc_w: float
    gamma_r: float
    efficiency_pct: float
    bifacial: bool
    technology: str
    source: str


@dataclass(frozen=True)
class InverterSpec:
    name: str
    paco_w: float
    pdco_w: float
    nominal_efficiency: float
    source: str


def _finite_positive(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric.") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{field} must be greater than zero.")
    return result


def _normalize_gamma(value: Any) -> float:
    """Return gamma_r as fraction per °C."""
    gamma = float(value)
    if not math.isfinite(gamma):
        raise ValueError("gamma_r must be finite.")

    # Supplied custom data uses values such as -0.30, meaning -0.30 %/°C.
    # Convert those percentages to fractions. Values already around -0.003
    # are assumed to already be fractional.
    return gamma / 100.0 if abs(gamma) > 0.05 else gamma


def _infer_bifacial(name: str) -> bool:
    return "bifacial" in name.lower() or "g2g" in name.lower()


def _infer_technology(name: str) -> str:
    lower = name.lower()
    if "topcon" in lower:
        return "TOPCon"
    if "perc" in lower:
        return "PERC"
    if "n_type" in lower or "n-type" in lower:
        return "N-type"
    if "mono" in lower:
        return "Mono"
    return "CEC/Manufacturer"


def _panel_from_row(name: str, row: pd.Series, source: str) -> PanelSpec:
    stc_w = _finite_positive(row.get("STC_W", row.get("STC_Watts", 0)), "STC_W")
    gamma_r = _normalize_gamma(row.get("gamma_r", -0.0032))

    try:
        eff = float(row.get("Efficiency", 20.5))
    except (TypeError, ValueError):
        eff = 20.5

    if not math.isfinite(eff) or eff <= 0:
        eff = 20.5

    return PanelSpec(
        name=name,
        stc_w=stc_w,
        gamma_r=gamma_r,
        efficiency_pct=eff,
        bifacial=_infer_bifacial(name),
        technology=_infer_technology(name),
        source=source,
    )


def _inverter_from_row(name: str, row: pd.Series, source: str) -> InverterSpec:
    paco_w = _finite_positive(row.get("Paco", 0), "Paco")
    pdco_raw = row.get("Pdco", paco_w / 0.98)

    try:
        pdco_w = float(pdco_raw)
    except (TypeError, ValueError):
        pdco_w = paco_w / 0.98

    if not math.isfinite(pdco_w) or pdco_w <= 0:
        pdco_w = paco_w / 0.98

    if pdco_w < paco_w:
        pdco_w = paco_w

    eff = paco_w / pdco_w if pdco_w > 0 else 0.98

    return InverterSpec(
        name=name,
        paco_w=paco_w,
        pdco_w=pdco_w,
        nominal_efficiency=float(max(0.0, min(eff, 1.0))),
        source=source,
    )


# ---------------------------------------------------------------------------
# Catalog loading
# ---------------------------------------------------------------------------

def load_hardware_catalogs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Return only the supplied Indian/custom hardware catalogue.

    pvlib CEC databases are intentionally NOT loaded here.
    The hardware catalogue is independent of pvlib's external CEC database.
    """
    panels = INDIAN_CUSTOM_PANELS.copy()
    inverters = INDIAN_CUSTOM_INVERTERS.copy()

    return panels, inverters


HARDWARE_PANELS, HARDWARE_INVERTERS = load_hardware_catalogs()

# Backward-compatible names used by the existing application.
# These now contain ONLY the supplied custom/Indian catalogue.
CEC_MOD_T = HARDWARE_PANELS
CEC_INV_T = HARDWARE_INVERTERS


# ---------------------------------------------------------------------------
# Stable lookup API
# ---------------------------------------------------------------------------

def get_panel_spec(panel_name: str) -> Optional[PanelSpec]:
    """Return a normalized panel specification, or None for an unknown model."""
    if not panel_name:
        return None

    key = str(panel_name).strip()

    if key not in INDIAN_CUSTOM_PANELS.index:
        return None

    row = INDIAN_CUSTOM_PANELS.loc[key]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    try:
        return _panel_from_row(key, row, "custom_indian")
    except (TypeError, ValueError):
        return None


def get_inverter_spec(inverter_name: str) -> Optional[InverterSpec]:
    """Return a normalized inverter specification, or None for an unknown model."""
    if not inverter_name:
        return None

    key = str(inverter_name).strip()

    if key not in INDIAN_CUSTOM_INVERTERS.index:
        return None

    row = INDIAN_CUSTOM_INVERTERS.loc[key]
    if isinstance(row, pd.DataFrame):
        row = row.iloc[0]

    try:
        return _inverter_from_row(key, row, "custom_indian")
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Backward-compatible scalar accessors for the new forecast engine
# ---------------------------------------------------------------------------

def get_panel_specs(panel_name: str) -> Tuple[float, float, float]:
    """
    Return (STC watts, gamma_r fraction/°C, efficiency percentage).

    Unknown models return None-like failure through ValueError rather than
    silently turning an invalid selection into a fictional 540 W panel.
    """
    spec = get_panel_spec(panel_name)
    if spec is None:
        raise KeyError(f"Unknown panel model: {panel_name}")
    return spec.stc_w, spec.gamma_r, spec.efficiency_pct


def get_inverter_specs(inverter_name: str) -> Tuple[float, float]:
    """
    Return (Paco watts, nominal DC->AC efficiency).

    This is only the nominal conversion parameter. Detailed inverter clipping
    is handled by the forecast/physics layer.
    """
    spec = get_inverter_spec(inverter_name)
    if spec is None:
        raise KeyError(f"Unknown inverter model: {inverter_name}")
    return spec.paco_w, spec.nominal_efficiency


# ---------------------------------------------------------------------------
# Search/list helpers used by the UI
# ---------------------------------------------------------------------------

def _normalized_search_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def search_panels(query: str = "", limit: int = 40) -> List[Dict[str, Any]]:
    query_n = _normalized_search_text(query)
    limit = max(1, min(int(limit), 200))

    results: List[Dict[str, Any]] = []
    for name in INDIAN_CUSTOM_PANELS.index.astype(str):
        if query_n and query_n not in _normalized_search_text(name):
            continue

        spec = get_panel_spec(name)
        if spec is None:
            continue

        results.append(
            {
                "id": name,
                "label": f"{name} ({spec.stc_w:.0f}W)",
                "power_w": round(spec.stc_w, 3),
                "stc_w": round(spec.stc_w, 3),
                "gamma_r": spec.gamma_r,
                "efficiency_pct": spec.efficiency_pct,
                "bifacial": spec.bifacial,
                "technology": spec.technology,
                "source": spec.source,
            }
        )

        if len(results) >= limit:
            break

    return results


def search_inverters(query: str = "", limit: int = 40) -> List[Dict[str, Any]]:
    query_n = _normalized_search_text(query)
    limit = max(1, min(int(limit), 200))

    results: List[Dict[str, Any]] = []
    for name in INDIAN_CUSTOM_INVERTERS.index.astype(str):
        if query_n and query_n not in _normalized_search_text(name):
            continue

        spec = get_inverter_spec(name)
        if spec is None:
            continue

        results.append(
            {
                "id": name,
                "label": f"{name} ({spec.paco_w:.0f}W AC)",
                "paco_w": round(spec.paco_w, 3),
                "pdco_w": round(spec.pdco_w, 3),
                "nominal_efficiency": round(spec.nominal_efficiency, 6),
                "source": spec.source,
            }
        )

        if len(results) >= limit:
            break

    return results


def catalog_summary() -> Dict[str, int]:
    return {
        "panel_count": int(len(INDIAN_CUSTOM_PANELS)),
        "inverter_count": int(len(INDIAN_CUSTOM_INVERTERS)),
        "custom_panel_count": int(len(INDIAN_CUSTOM_PANELS)),
        "custom_inverter_count": int(len(INDIAN_CUSTOM_INVERTERS)),
    }


__all__ = [
    "PanelSpec",
    "InverterSpec",
    "INDIAN_CUSTOM_PANELS",
    "INDIAN_CUSTOM_INVERTERS",
    "HARDWARE_PANELS",
    "HARDWARE_INVERTERS",
    "CEC_MOD_T",
    "CEC_INV_T",
    "load_hardware_catalogs",
    "get_panel_spec",
    "get_inverter_spec",
    "get_panel_specs",
    "get_inverter_specs",
    "search_panels",
    "search_inverters",
    "catalog_summary",
]
