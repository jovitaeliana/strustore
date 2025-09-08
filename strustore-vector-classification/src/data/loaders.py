"""
Data loading utilities for the vector classification system.
"""
import pandas as pd
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json

from ..models.schemas import GamingConsoleItem
from ..core.exceptions import ProcessingError
from ..config.logging import log_performance

logger = logging.getLogger(__name__)


class MasterItemsLoader:
    """Load and process master items data for vector database population."""
    
    def __init__(self, 
                 master_list_path: str,
                 items_prompt_path: Optional[str] = None):
        """
        Initialize the loader.
        
        Args:
            master_list_path: Path to master-list.csv
            items_prompt_path: Optional path to items-prompt.csv
        """
        self.master_list_path = Path(master_list_path)
        self.items_prompt_path = Path(items_prompt_path) if items_prompt_path else None
        
        self._validate_paths()
        
        # Data storage
        self.master_df = None
        self.prompts_df = None
        self.processed_items = []
    
    def _validate_paths(self):
        """Validate that required files exist."""
        if not self.master_list_path.exists():
            raise ProcessingError(f"Master list file not found: {self.master_list_path}")
        
        if self.items_prompt_path and not self.items_prompt_path.exists():
            raise ProcessingError(f"Items prompt file not found: {self.items_prompt_path}")
        
        logger.info(f"Validated paths - Master: {self.master_list_path}")
    
    @log_performance
    def load_master_items(self) -> List[GamingConsoleItem]:
        """
        Load and process master items.
        
        Returns:
            List of GamingConsoleItem objects
        """
        try:
            # Load master list CSV
            self.master_df = pd.read_csv(self.master_list_path)
            logger.info(f"Loaded {len(self.master_df)} items from master list")
            
            # Validate required columns
            required_columns = ['id', 'item']
            self._validate_columns(self.master_df, required_columns, "master list")
            
            # Load prompts if available
            prompts_dict = {}
            if self.items_prompt_path:
                self.prompts_df = pd.read_csv(self.items_prompt_path)
                prompts_dict = dict(zip(
                    self.prompts_df['id'].astype(str),
                    self.prompts_df['prompt']
                ))
                logger.info(f"Loaded {len(prompts_dict)} prompts")
            
            # Process items
            processed_items = []
            for _, row in self.master_df.iterrows():
                try:
                    item = self._process_single_item(row, prompts_dict)
                    if item:
                        processed_items.append(item)
                except Exception as e:
                    logger.error(f"Failed to process item {row.get('id', 'unknown')}: {e}")
                    continue
            
            self.processed_items = processed_items
            logger.info(f"Successfully processed {len(processed_items)} items")
            
            return processed_items
            
        except Exception as e:
            raise ProcessingError(f"Failed to load master items: {e}")
    
    def _validate_columns(self, df: pd.DataFrame, required_columns: List[str], file_type: str):
        """Validate that DataFrame has required columns."""
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ProcessingError(f"{file_type} missing required columns: {missing_columns}")
    
    def _process_single_item(self, 
                           row: pd.Series, 
                           prompts_dict: Dict[str, str]) -> Optional[GamingConsoleItem]:
        """
        Process a single item row into a GamingConsoleItem.
        
        Args:
            row: DataFrame row with item data
            prompts_dict: Dictionary mapping item ID to prompts
        
        Returns:
            GamingConsoleItem or None if processing fails
        """
        try:
            item_id = str(row['id'])
            item_name = str(row['item']).strip()
            
            # Skip if name is empty or placeholder
            if not item_name or item_name.lower() in ['', 'nan', 'null', 'none']:
                logger.warning(f"Skipping item {item_id} with empty name")
                return None
            
            # Extract gaming console information
            category = self._extract_category(item_name)
            manufacturer = self._extract_manufacturer(item_name)
            device_type = self._extract_device_type(item_name)
            model_codes = self._extract_model_codes(item_name)
            synonyms = self._generate_synonyms(item_name)
            
            # Get description from prompts if available
            description = prompts_dict.get(item_id, "")
            
            # Create gaming console item
            gaming_item = GamingConsoleItem(
                id=item_id,
                name=item_name,
                category=category,
                manufacturer=manufacturer,
                device_type=device_type,
                model_codes=model_codes,
                synonyms=synonyms,
                description=description
            )
            
            logger.debug(f"Processed item: {item_id} - {item_name}")
            return gaming_item
            
        except Exception as e:
            logger.error(f"Error processing item row: {e}")
            return None
    
    def _extract_category(self, item_name: str) -> Optional[str]:
        """Extract category from item name."""
        name_lower = item_name.lower()
        
        if 'console' in name_lower:
            if any(term in name_lower for term in ['handheld', 'ds', 'gameboy', 'psp', 'vita']):
                return 'handheld_console'
            else:
                return 'home_console'
        elif 'controller' in name_lower:
            return 'controller'
        elif 'accessory' in name_lower:
            return 'accessory'
        else:
            return 'gaming_hardware'
    
    def _extract_manufacturer(self, item_name: str) -> str:
        """Extract manufacturer from item name."""
        name_lower = item_name.lower()
        
        if 'nintendo' in name_lower:
            return 'Nintendo'
        elif any(term in name_lower for term in ['sony', 'playstation']):
            return 'Sony'
        elif any(term in name_lower for term in ['microsoft', 'xbox']):
            return 'Microsoft'
        elif 'sega' in name_lower:
            return 'Sega'
        elif 'atari' in name_lower:
            return 'Atari'
        else:
            return 'Nintendo'  # Default for this dataset
    
    def _extract_device_type(self, item_name: str) -> Optional[str]:
        """Extract device type from item name."""
        name_lower = item_name.lower()
        
        if any(term in name_lower for term in ['ds', 'gameboy', 'game boy', 'handheld']):
            return 'handheld'
        elif 'console' in name_lower and 'handheld' not in name_lower:
            return 'console'
        elif 'controller' in name_lower:
            return 'controller'
        else:
            return 'console'  # Default assumption
    
    def _extract_model_codes(self, item_name: str) -> List[str]:
        """Extract model codes from item name."""
        import re
        
        model_codes = []
        
        # Common Nintendo model code patterns
        patterns = [
            r'\b[A-Z]{3}-\d{3}\b',  # NTR-001, AGS-001, etc.
            r'\b[A-Z]{2,4}\d{3,4}\b',  # AGB001, DOL001, etc.
            r'\b\d{4}\b'  # Year codes like 2004, 2006
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, item_name)
            model_codes.extend(matches)
        
        # Specific model mappings
        name_lower = item_name.lower()
        if 'ds original' in name_lower or 'ds console' in name_lower:
            model_codes.append('NTR-001')
        elif 'ds lite' in name_lower:
            model_codes.append('USG-001')
        elif 'dsi' in name_lower:
            model_codes.append('TWL-001')
        elif '3ds' in name_lower and 'xl' not in name_lower:
            model_codes.append('CTR-001')
        elif '2ds' in name_lower:
            model_codes.append('FTR-001')
        
        return list(set(model_codes))  # Remove duplicates
    
    def _generate_synonyms(self, item_name: str) -> List[str]:
        """Generate synonyms and alternative names."""
        synonyms = []
        name_lower = item_name.lower()
        
        # Nintendo DS variations
        if 'nintendo ds' in name_lower:
            synonyms.extend([
                'DS console', 'Nintendo DS system', 'Dual Screen',
                'NDS', 'Nintendo handheld'
            ])
            
            if 'original' in name_lower:
                synonyms.extend(['DS Phat', 'Original DS', 'First DS'])
            elif 'lite' in name_lower:
                synonyms.extend(['DS Lite', 'DSL', 'Slim DS'])
        
        # Game Boy variations
        if 'game boy' in name_lower or 'gameboy' in name_lower:
            synonyms.extend([
                'GB', 'Nintendo Game Boy', 'Portable Nintendo',
                'Handheld gaming device'
            ])
            
            if 'advance' in name_lower:
                synonyms.extend(['GBA', 'Game Boy Advance'])
            elif 'micro' in name_lower:
                synonyms.extend(['GBM', 'Game Boy Micro'])
        
        # Generic gaming terms
        if 'console' in name_lower:
            synonyms.extend([
                'gaming system', 'video game console',
                'gaming hardware', 'game machine'
            ])
        
        if 'handheld' in name_lower:
            synonyms.extend([
                'portable console', 'mobile gaming',
                'pocket gaming', 'portable gaming device'
            ])
        
        # Remove duplicates and original name
        synonyms = [s for s in set(synonyms) if s.lower() != item_name.lower()]
        
        return synonyms
    
    def get_items_for_vectorization(self) -> List[Dict[str, Any]]:
        """
        Get items formatted for vector database insertion.
        
        Returns:
            List of dictionaries ready for vectorization
        """
        if not self.processed_items:
            raise ProcessingError("No items loaded. Call load_master_items() first.")
        
        vectorization_data = []
        
        for item in self.processed_items:
            # Create embedding text
            embedding_text = item.to_embedding_text()
            
            # Create metadata
            metadata = {
                'item_id': item.id,
                'item_name': item.name,
                'category': item.category,
                'manufacturer': item.manufacturer,
                'device_type': item.device_type,
                'model_codes': item.model_codes,
                'synonyms': item.synonyms,
                'text': embedding_text,  # This will be the document text
                'description': item.description or ""
            }
            
            vectorization_data.append({
                'id': item.id,
                'text': embedding_text,
                'metadata': metadata
            })
        
        logger.info(f"Prepared {len(vectorization_data)} items for vectorization")
        return vectorization_data
    
    def export_processed_items(self, output_path: str, format: str = 'json'):
        """
        Export processed items to file.
        
        Args:
            output_path: Output file path
            format: Export format ('json' or 'csv')
        """
        if not self.processed_items:
            raise ProcessingError("No items to export. Load items first.")
        
        output_path = Path(output_path)
        
        try:
            if format.lower() == 'json':
                # Export as JSON
                data = [item.dict() for item in self.processed_items]
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                    
            elif format.lower() == 'csv':
                # Export as CSV (flattened)
                rows = []
                for item in self.processed_items:
                    row = {
                        'id': item.id,
                        'name': item.name,
                        'category': item.category,
                        'manufacturer': item.manufacturer,
                        'device_type': item.device_type,
                        'model_codes': '; '.join(item.model_codes),
                        'synonyms': '; '.join(item.synonyms),
                        'description': item.description,
                        'embedding_text': item.to_embedding_text()
                    }
                    rows.append(row)
                
                df = pd.DataFrame(rows)
                df.to_csv(output_path, index=False)
            
            else:
                raise ProcessingError(f"Unsupported export format: {format}")
            
            logger.info(f"Exported {len(self.processed_items)} items to {output_path}")
            
        except Exception as e:
            raise ProcessingError(f"Failed to export items: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded items."""
        if not self.processed_items:
            return {"error": "No items loaded"}
        
        # Count by category
        categories = {}
        manufacturers = {}
        device_types = {}
        
        for item in self.processed_items:
            categories[item.category] = categories.get(item.category, 0) + 1
            manufacturers[item.manufacturer] = manufacturers.get(item.manufacturer, 0) + 1
            device_types[item.device_type] = device_types.get(item.device_type, 0) + 1
        
        # Calculate text statistics
        text_lengths = [len(item.to_embedding_text()) for item in self.processed_items]
        
        return {
            "total_items": len(self.processed_items),
            "categories": dict(sorted(categories.items())),
            "manufacturers": dict(sorted(manufacturers.items())),
            "device_types": dict(sorted(device_types.items())),
            "text_length_stats": {
                "min": min(text_lengths) if text_lengths else 0,
                "max": max(text_lengths) if text_lengths else 0,
                "avg": sum(text_lengths) / len(text_lengths) if text_lengths else 0
            },
            "items_with_model_codes": sum(1 for item in self.processed_items if item.model_codes),
            "items_with_synonyms": sum(1 for item in self.processed_items if item.synonyms),
            "items_with_descriptions": sum(1 for item in self.processed_items if item.description)
        }


class ItemsJSONLoader:
    """Load hierarchical items data from items.json."""
    
    def __init__(self, items_json_path: str):
        """
        Initialize JSON loader.
        
        Args:
            items_json_path: Path to items.json file
        """
        self.items_json_path = Path(items_json_path)
        self.device_families = []
        
        if not self.items_json_path.exists():
            raise ProcessingError(f"Items JSON file not found: {self.items_json_path}")
    
    @log_performance
    def load_device_families(self) -> List[Dict[str, Any]]:
        """
        Load device families from JSON.
        
        Returns:
            List of device family dictionaries
        """
        try:
            with open(self.items_json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            self.device_families = data.get('DeviceFamilies', [])
            logger.info(f"Loaded {len(self.device_families)} device families")
            
            return self.device_families
            
        except Exception as e:
            raise ProcessingError(f"Failed to load items JSON: {e}")
    
    def get_flat_items_list(self) -> List[Dict[str, Any]]:
        """
        Get flattened list of all items from families.
        
        Returns:
            List of individual items
        """
        if not self.device_families:
            self.load_device_families()
        
        flat_items = []
        
        for family in self.device_families:
            family_name = family.get('family', 'Unknown Family')
            family_id = family.get('id', 0)
            
            children = family.get('children', [])
            for i, child in enumerate(children):
                if child.get('ignore', False):
                    continue
                
                item = {
                    'id': f"{family_id}_{i}",
                    'name': child.get('name', ''),
                    'family': family_name,
                    'family_id': family_id,
                    'code': child.get('code'),
                    'search': child.get('search')
                }
                
                flat_items.append(item)
        
        logger.info(f"Generated {len(flat_items)} flat items from families")
        return flat_items