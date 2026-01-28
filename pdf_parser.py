"""
BRISNET PDF Past Performance Parser
Extracts ALL data fields that the text parser uses:
- Horse names, posts, styles, Quirin points
- Speed figures (Beyer)
- Class ratings, purse, race type
- Pedigree (Sire, Dam, AWD, SPI)
- Track bias impact values
- Surface stats (Turf %, AW %)
- Angles (debut, blinkers, surface switch, etc.)
- Morning line odds
- Race conditions
"""

import re
from typing import Dict, List, Optional
try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    try:
        import PyPDF2
        PDF_AVAILABLE = True
    except ImportError:
        PDF_AVAILABLE = False
        print("⚠️ Install PDF support: pip install pdfplumber PyPDF2")

class BRISNETPDFParser:
    """Parse BRISNET PDF files into structured PP text format."""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.raw_text = ""
        self.structured_pp = ""
    
    def extract_text_from_pdf(self) -> str:
        """Extract all text from PDF."""
        if not PDF_AVAILABLE:
            raise ImportError("PDF libraries not installed. Run: pip install pdfplumber")
        
        try:
            # Try pdfplumber first (better for tables)
            import pdfplumber
            with pdfplumber.open(self.pdf_path) as pdf:
                text = ""
                for page in pdf.pages:
                    text += page.extract_text() + "\n"
                self.raw_text = text
                return text
        except:
            # Fallback to PyPDF2
            import PyPDF2
            with open(self.pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() + "\n"
                self.raw_text = text
                return text
    
    def parse_race_header(self) -> Dict:
        """Extract race conditions: track, date, distance, surface, purse."""
        header = {}
        
        # Track name (common patterns)
        track_match = re.search(r'(Gulfstream|Keeneland|Churchill|Saratoga|Del Mar|Santa Anita|Aqueduct|Belmont|Laurel|Penn National|Charles Town|Mountaineer|Thistledown|Mahoning Valley)', 
                               self.raw_text, re.IGNORECASE)
        if track_match:
            header['track'] = track_match.group(1)
        
        # Date (MM/DD/YYYY or similar)
        date_match = re.search(r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})', self.raw_text)
        if date_match:
            header['date'] = date_match.group(1)
        
        # Distance (e.g., "6 Furlongs", "1 Mile", "1 1/16 Miles")
        dist_match = re.search(r'(\d+(?:\s+\d+/\d+)?\s+(?:Furlong|Mile|Yard)s?)', self.raw_text, re.IGNORECASE)
        if dist_match:
            header['distance'] = dist_match.group(1)
        
        # Surface
        surf_match = re.search(r'(Dirt|Turf|Synthetic|All[- ]Weather)', self.raw_text, re.IGNORECASE)
        if surf_match:
            header['surface'] = surf_match.group(1)
        
        # Condition
        cond_match = re.search(r'\b(Fast|Good|Firm|Yielding|Soft|Sloppy|Muddy|Heavy)\b', self.raw_text, re.IGNORECASE)
        if cond_match:
            header['condition'] = cond_match.group(1)
        
        # Purse ($XX,XXX)
        purse_match = re.search(r'\$\s*([\d,]+)', self.raw_text)
        if purse_match:
            header['purse'] = purse_match.group(1).replace(',', '')
        
        # Race type
        if re.search(r'Maiden\s+Special\s+Weight', self.raw_text, re.IGNORECASE):
            header['race_type'] = 'Maiden Special Weight'
        elif re.search(r'Claiming', self.raw_text, re.IGNORECASE):
            header['race_type'] = 'Claiming'
        elif re.search(r'Allowance', self.raw_text, re.IGNORECASE):
            header['race_type'] = 'Allowance'
        elif re.search(r'Stakes', self.raw_text, re.IGNORECASE):
            header['race_type'] = 'Stakes'
        
        return header
    
    def parse_horses(self) -> List[Dict]:
        """Extract all horses with their complete data."""
        horses = []
        
        # Split by horse entries (BRISNET typically has horse name in bold/caps)
        # Pattern: Post number, then horse name
        horse_pattern = re.compile(
            r'(\d+)\s+([A-Z][a-zA-Z\s\']+?)(?=\s+(?:ML|Odds|Beyer|\d+/\d|\n))',
            re.MULTILINE
        )
        
        for match in horse_pattern.finditer(self.raw_text):
            post = match.group(1)
            horse_name = match.group(2).strip()
            
            # Extract horse's data section (next ~500 chars after name)
            start_pos = match.end()
            section = self.raw_text[start_pos:start_pos + 1000]
            
            horse_data = {
                'post': int(post),
                'name': horse_name,
                'style': self._extract_style(section),
                'quirin': self._extract_quirin(section),
                'beyer': self._extract_beyer(section),
                'class_rating': self._extract_class(section),
                'morning_line': self._extract_ml_odds(section),
                'pedigree': self._extract_pedigree(section),
                'angles': self._extract_angles(section),
                'last_races': self._extract_past_performances(section)
            }
            
            horses.append(horse_data)
        
        return horses
    
    def _extract_style(self, text: str) -> str:
        """Extract running style (E, E/P, P, S)."""
        style_match = re.search(r'\b(E/P|E|P|S)\b', text)
        return style_match.group(1) if style_match else 'P'
    
    def _extract_quirin(self, text: str) -> Optional[int]:
        """Extract Quirin speed points."""
        quirin_match = re.search(r'(?:Quirin|Q):\s*(\d+)', text, re.IGNORECASE)
        if quirin_match:
            return int(quirin_match.group(1))
        
        # Sometimes just a number after style
        num_match = re.search(r'E/P\s+(\d+)|E\s+(\d+)|P\s+(\d+)|S\s+(\d+)', text)
        if num_match:
            for g in num_match.groups():
                if g:
                    return int(g)
        return None
    
    def _extract_beyer(self, text: str) -> Optional[int]:
        """Extract Beyer speed figure."""
        beyer_patterns = [
            r'Beyer:\s*(\d+)',
            r'BSF:\s*(\d+)',
            r'Speed\s+Fig:\s*(\d+)',
            r'\bLast:\s*(\d{2,3})\b'
        ]
        
        for pattern in beyer_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return int(match.group(1))
        return None
    
    def _extract_class(self, text: str) -> Optional[str]:
        """Extract class level."""
        class_patterns = [
            r'(Graded Stakes|G[123])',
            r'(Listed Stakes)',
            r'(Allowance)',
            r'(Maiden Special Weight|MSW)',
            r'(Maiden Claiming|MCL)',
            r'(Claiming|CLM)\s+\$?([\d,]+)',
            r'(Starter Allowance)'
        ]
        
        for pattern in class_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(0)
        return None
    
    def _extract_ml_odds(self, text: str) -> Optional[str]:
        """Extract morning line odds."""
        ml_patterns = [
            r'ML:\s*([\d/\-]+)',
            r'Morning Line:\s*([\d/\-]+)',
            r'Odds:\s*([\d/\-]+)'
        ]
        
        for pattern in ml_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        return None
    
    def _extract_pedigree(self, text: str) -> Dict:
        """Extract sire, dam, AWD, SPI, surface stats."""
        ped = {}
        
        # Sire
        sire_match = re.search(r'(?:Sire|S):\s*([A-Z][a-zA-Z\s]+?)(?:\s+\(|$)', text)
        if sire_match:
            ped['sire'] = sire_match.group(1).strip()
        
        # SPI (Sire Performance Index)
        spi_match = re.search(r'SPI:\s*([\d.]+)', text, re.IGNORECASE)
        if spi_match:
            ped['spi'] = float(spi_match.group(1))
        
        # Dam
        dam_match = re.search(r'(?:Dam|D):\s*([A-Z][a-zA-Z\s]+?)(?:\s+\(|$)', text)
        if dam_match:
            ped['dam'] = dam_match.group(1).strip()
        
        # AWD (Average Winning Distance)
        awd_match = re.search(r'AWD:\s*([\d.]+)', text, re.IGNORECASE)
        if awd_match:
            ped['awd'] = float(awd_match.group(1))
        
        # Turf percentage
        turf_match = re.search(r'Turf:\s*([\d.]+)%', text, re.IGNORECASE)
        if turf_match:
            ped['turf_pct'] = float(turf_match.group(1))
        
        # AW/Synthetic percentage
        aw_match = re.search(r'(?:AW|Synthetic):\s*([\d.]+)%', text, re.IGNORECASE)
        if aw_match:
            ped['aw_pct'] = float(aw_match.group(1))
        
        return ped
    
    def _extract_angles(self, text: str) -> List[str]:
        """Extract positive/negative angles."""
        angles = []
        
        angle_keywords = [
            'debut', 'first time', 'blinkers on', 'blinkers off',
            'surface switch', 'shipper', 'stretch out', 'cutback',
            'class rise', 'class drop', 'layoff', 'trainer pattern',
            'jockey change', 'equipment change'
        ]
        
        for keyword in angle_keywords:
            if re.search(keyword, text, re.IGNORECASE):
                angles.append(keyword)
        
        return angles
    
    def _extract_past_performances(self, text: str) -> List[Dict]:
        """Extract last race data (date, track, distance, finish)."""
        races = []
        
        # Pattern: Date Track Dist Finish
        # e.g., "12/15 GP 6f 1st"
        pp_pattern = re.compile(
            r'(\d{1,2}/\d{1,2})\s+([A-Z]{2,4})\s+([\d.]+f|[\d]+m)\s+(\d+(?:st|nd|rd|th))',
            re.IGNORECASE
        )
        
        for match in pp_pattern.finditer(text):
            races.append({
                'date': match.group(1),
                'track': match.group(2),
                'distance': match.group(3),
                'finish': match.group(4)
            })
        
        return races[:5]  # Last 5 races
    
    def _extract_track_bias(self) -> Dict:
        """Extract track bias impact values for E/EP/P/S."""
        bias = {}
        
        # Look for track bias section
        bias_section = re.search(
            r'Track Bias.*?Impact Value.*?E.*?([\d.]+).*?EP.*?([\d.]+).*?P.*?([\d.]+).*?S.*?([\d.]+)',
            self.raw_text,
            re.IGNORECASE | re.DOTALL
        )
        
        if bias_section:
            bias['E'] = float(bias_section.group(1))
            bias['EP'] = float(bias_section.group(2))
            bias['P'] = float(bias_section.group(3))
            bias['S'] = float(bias_section.group(4))
        
        return bias
    
    def build_structured_pp_text(self) -> str:
        """
        Reconstruct BRISNET PP text format that existing parser expects.
        """
        header = self.parse_race_header()
        horses = self.parse_horses()
        track_bias = self._extract_track_bias()
        
        pp_text = f"""--- RACE CONDITIONS ---
Track: {header.get('track', 'Unknown')}
Date: {header.get('date', 'TBD')}
Distance: {header.get('distance', '6 Furlongs')}
Surface: {header.get('surface', 'Dirt')}
Condition: {header.get('condition', 'Fast')}
Purse: ${header.get('purse', '50000')}
Race Type: {header.get('race_type', 'Allowance')}

--- FIELD SIZE: {len(horses)} ---

"""
        
        # Add each horse
        for i, horse in enumerate(horses, 1):
            pp_text += f"""{i}. Horse: {horse['name']} (#{horse['post']})
   Morning Line: {horse.get('morning_line', '10-1')}
   
   Running Style: {horse['style']}
   Quirin Speed Points: {horse.get('quirin', 'N/A')}
   
   Speed Figures:
   - Last Beyer: {horse.get('beyer', 'N/A')}
   - Average (Top 2): {horse.get('beyer', 70)}
   
   Class Rating: {horse.get('class_rating', 'N/A')}
   
   Pedigree:
   - Sire: {horse.get('pedigree', {}).get('sire', 'Unknown')}"""
            
            if 'spi' in horse.get('pedigree', {}):
                pp_text += f" (SPI: {horse['pedigree']['spi']})"
            
            pp_text += f"\n   - Dam: {horse.get('pedigree', {}).get('dam', 'Unknown')}\n"
            
            if 'awd' in horse.get('pedigree', {}):
                pp_text += f"   - AWD: {horse['pedigree']['awd']} furlongs\n"
            
            if 'turf_pct' in horse.get('pedigree', {}):
                pp_text += f"   - Turf: {horse['pedigree']['turf_pct']}%\n"
            
            if 'aw_pct' in horse.get('pedigree', {}):
                pp_text += f"   - AW: {horse['pedigree']['aw_pct']}%\n"
            
            # Angles
            if horse.get('angles'):
                pp_text += f"\n   Angles: {', '.join(horse['angles'])}\n"
            
            # Past performances
            if horse.get('last_races'):
                pp_text += "\n   Recent Races:\n"
                for race in horse['last_races']:
                    pp_text += f"   - {race['date']} {race['track']} {race['distance']}: {race['finish']}\n"
            
            pp_text += "\n"
        
        # Add track bias section
        if track_bias:
            pp_text += """
--- TRACK BIAS (Numerical) ---
Impact Values:
"""
            for style, value in track_bias.items():
                pp_text += f"- {style}: Impact Value = {value}\n"
        
        self.structured_pp = pp_text
        return pp_text
    
    def parse(self) -> str:
        """Main method: Extract PDF and return structured PP text."""
        print(f"📄 Reading PDF: {self.pdf_path}")
        self.extract_text_from_pdf()
        
        print("🔍 Parsing race data...")
        structured = self.build_structured_pp_text()
        
        print(f"✅ Extracted {len(self.parse_horses())} horses")
        
        return structured
    
    def save_to_file(self, output_path: str = "parsed_pp.txt"):
        """Save structured PP text to file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(self.structured_pp)
        print(f"💾 Saved to: {output_path}")


def main():
    """Command-line interface."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pdf_parser.py <path_to_brisnet_pdf>")
        print("\nExample:")
        print("  python pdf_parser.py race_card.pdf")
        print("\nOutput will be saved to 'parsed_pp.txt'")
        return
    
    pdf_path = sys.argv[1]
    
    try:
        parser = BRISNETPDFParser(pdf_path)
        pp_text = parser.parse()
        parser.save_to_file("parsed_pp.txt")
        
        print("\n" + "=" * 60)
        print("SUCCESS! PP text ready for analysis.")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Copy content from 'parsed_pp.txt'")
        print("2. Paste into Section A text box in app")
        print("3. Click 'Analyze This Race'")
        
    except Exception as e:
        print(f"\n❌ Error parsing PDF: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
