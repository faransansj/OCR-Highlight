import os
import logging
from typing import List, Dict, Optional
from notion_client import Client

logger = logging.getLogger("NotionExporter")

class NotionExporter:
    """
    Exporter to send OCR results to a Notion Database.
    """
    def __init__(self, auth_token: Optional[str] = None):
        self.token = auth_token or os.environ.get("NOTION_TOKEN")
        if not self.token:
            logger.error("Notion token not found in environment.")
            self.client = None
        else:
            self.client = Client(auth=self.token)
            logger.info("Notion client initialized.")

    def convert_to_rich_text(self, results: List[Dict]) -> List[Dict]:
        """
        Convert structured OCR results to Notion Rich Text blocks.
        Maps highlights and underlines to Notion's annotation styles.
        """
        rich_text_blocks = []
        
        # Color mapping: Notion colors end with _background for highlights
        NOTION_COLORS = {
            "yellow": "yellow_background",
            "green": "green_background",
            "pink": "pink_background",
            "blue": "blue_background",
            "unknown": "gray_background"
        }

        for res in results:
            text = res.get('text', '')
            if not text:
                continue
            
            m_type = res.get('markup_type', '')
            subtype = res.get('subtype', 'standard')
            
            # Base annotation object
            annotations = {
                "bold": False,
                "italic": False,
                "strikethrough": False,
                "underline": False,
                "code": False,
                "color": "default"
            }
            
            if m_type == 'highlight':
                annotations["color"] = NOTION_COLORS.get(subtype, "yellow_background")
            elif m_type == 'underline':
                annotations["underline"] = True
            elif m_type == 'strikethrough':
                annotations["strikethrough"] = True
            elif m_type in ['circle', 'rectangle']:
                annotations["bold"] = True # Highlight structural marks with bold
                annotations["color"] = "blue"

            rich_text_blocks.append({
                "type": "text",
                "text": {"content": text + " "}, # Add space between fragments
                "annotations": annotations
            })
            
        return rich_text_blocks

    def prepare_database_properties(self, title: str, source_path: str, tags: List[str] = None) -> Dict:
        """
        Prepare standard database properties for entry.
        Assumes the database has 'Name', 'Source', and 'Tags' columns.
        """
        properties = {
            "Name": {
                "title": [{"text": {"content": title}}]
            },
            "Source": {
                "rich_text": [{"text": {"content": os.path.basename(source_path)}}]
            }
        }
        
        if tags:
            properties["Tags"] = {
                "multi_select": [{"name": tag} for tag in tags]
            }
            
        return properties

    def markdown_to_blocks(self, markdown_text: str) -> List[Dict]:
        """
        Simple converter from Markdown text to Notion blocks.
        Handles headings and paragraphs.
        """
        blocks = []
        lines = markdown_text.split("\n")
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("# "):
                blocks.append({
                    "object": "block",
                    "type": "heading_1",
                    "heading_1": {"rich_text": [{"type": "text", "text": {"content": line[2:]}}]}
                })
            elif line.startswith("## "):
                blocks.append({
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {"rich_text": [{"type": "text", "text": {"content": line[3:]}}]}
                })
            elif line.startswith("- ") or line.startswith("* "):
                blocks.append({
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {"rich_text": [{"type": "text", "text": {"content": line[2:]}}]}
                })
            else:
                # Regular paragraph
                blocks.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": [{"type": "text", "text": {"content": line}}]}
                })
                
        return blocks

    def create_page(self, database_id: str, results: Dict, properties: Optional[Dict] = None):
        """
        Create a new page in the specified database with Rich Text mapping.
        'results' should be the dict returned by process_image.
        """
        if not self.client:
            logger.error("Notion client not initialized. Cannot create page.")
            return None

        image_path = results.get('image_path', 'unknown_source.png')
        title = os.path.basename(image_path)
        
        # Convert results to Notion's rich_text format
        rich_text_content = self.convert_to_rich_text(results.get('results', []))
        
        # Prepare properties
        if properties is None:
            # Detect unique markup types for tags
            found_types = list(set([r.get('markup_type') for r in results.get('results', [])]))
            properties = self.prepare_database_properties(title, image_path, tags=found_types)

        try:
            new_page = self.client.pages.create(
                parent={"database_id": database_id},
                properties=properties,
                children=[
                    {
                        "object": "block",
                        "type": "heading_2",
                        "heading_2": {"rich_text": [{"type": "text", "text": {"content": "Extracted Knowledge (Rich Text)"}}]}
                    },
                    {
                        "object": "block",
                        "type": "paragraph",
                        "paragraph": {
                            "rich_text": rich_text_content[:100]
                        }
                    },
                    {
                        "object": "block",
                        "type": "callout",
                        "callout": {
                            "rich_text": [{"type": "text", "text": {"content": f"Processed via OCR-Highlight v2.0 Pipeline. Full analysis saved in local inventory."}}],
                            "icon": {"emoji": "🛡️"}
                        }
                    }
                ]
            )
            logger.info(f"Page created in Database: {new_page['id']}")
            return new_page
        except Exception as e:
            logger.error(f"Failed to create Notion page: {e}")
            return None

if __name__ == "__main__":
    # Test initialization
    exporter = NotionExporter()
    if exporter.token:
        print("Notion Exporter ready.")
    else:
        print("Notion Exporter missing credentials.")
