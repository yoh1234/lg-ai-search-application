from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import json
import time


from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager
import json
import time


class LGProductScraper:
    def __init__(self, category_url: str, category_name: str = "Unknown"):
        self.url = category_url
        self.category = category_name
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
        self.data = []

    def open_page(self):
        self.driver.get(self.url)

    def click_view_all(self):
        try:      
            # 방법 1: data-testid로 찾기
            toggle = WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, '[data-testid="toggle"]'))
            )
            
            # 현재 상태 확인
            is_checked = toggle.get_attribute("aria-checked") == "true"
            print(f"Current toggle status: {is_checked}")
            
            # 토글이 비활성화되어 있다면 클릭
            if not is_checked:
                toggle.click()
                print("View all toggle activated")
                
                # 토글 활성화 후 페이지 로딩 대기
                WebDriverWait(self.driver, 10).until(
                    EC.element_to_be_clickable((By.CSS_SELECTOR, '[aria-checked="true"]'))
                )

            
        except Exception as e:
            print(f"Error: {e}")
    
    def click_load_more(self):
        while True:
            try:
                load_more_button = WebDriverWait(self.driver, 3).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Load More')]"))
                )
                
                # 스크롤 없이 바로 클릭 (더 빠름)
                self.driver.execute_script("arguments[0].click();", load_more_button)
                print("load more button clicked")
                
                time.sleep(1)  # 1초면 충분
                # 이미지의 HTML 구조에 맞는 CSS 선택자로 수정
                # load_more_button = WebDriverWait(self.driver, 5).until(
                #     EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Load More')]"))
                # )
                
                # # 버튼이 화면에 보이도록 스크롤
                # self.driver.execute_script("arguments[0].scrollIntoView(true);", load_more_button)
                # time.sleep(1)
                
                # # 버튼 클릭
                # self.driver.execute_script("arguments[0].click();", load_more_button)
                # print("load more button has been clicked")
                # time.sleep(2)
                
            except:
                print(f"[{self.category}] All products loaded.")
                break

    def extract_product_info(self, card, size) -> dict:
        """카드에서 제품 정보 추출"""
        try:
            product_info = {
                'size': size,
                'product_url': self.get_product_url(card),
                'product_name': self.get_product_name(card),
                'sku': self.get_product_sku(card),
                'price': self.get_product_price(card)
            }
            return product_info
        except Exception as e:
            print(f"    정보 추출 실패: {e}")
            return None

    def get_product_url(self, card):
        """제품 URL 추출"""
        try:
            link = card.find_element(By.CSS_SELECTOR, "a[href*='/us/']").get_attribute('href')
            return link
        except:
            return "N/A"

    def get_product_name(self, card):
        """제품명 추출"""
        try:
            name_selectors = [
                ".product-card-title-expandable",
                "h3.MuiTypography-body1",
                ".mh-title h3"
            ]
            for selector in name_selectors:
                try:
                    name = card.find_element(By.CSS_SELECTOR, selector).text.strip()
                    if name:
                        return name
                except:
                    continue
            return "N/A"
        except:
            return "N/A"

    def get_product_sku(self, card):
        """SKU 추출"""
        try:
            sku_selectors = [
                ".MuiTypography-caption",
                ".css-1undsyy"
            ]
            for selector in sku_selectors:
                try:
                    sku = card.find_element(By.CSS_SELECTOR, selector).text.strip()
                    if sku and not sku.startswith('('):  # 리뷰 수가 아닌 것
                        return sku
                except:
                    continue
            return "N/A"
        except:
            return "N/A"

    def get_product_price(self, card) -> float:
        """가격 추출"""
        try:
            price_selectors = [
                "h4.MuiTypography-subtitle1",
                ".css-1h0f645",
                "[data-testid='pricesave'] h4"
            ]
            for selector in price_selectors:
                try:
                    price_element = card.find_element(By.CSS_SELECTOR, selector)
                    price_text = price_element.text.strip()
                    # "$2,199.99" -> "2199.99"
                    price_clean = price_text.replace('$', '').replace(',', '').split()[0]
                    return float(price_clean)
                except:
                    continue
            return "N/A"
        except:
            return "N/A"

    def scrape_product_details(self, products: list) -> list:
        """각 제품의 상세 페이지에서 추가 정보 수집"""
        updated_products = []
        print("products: ", products)
        for i, product in enumerate(products):
            print(f"제품 {i+1}/{len(products)} 상세 정보 수집 중...")
            print(f"URL: {product['product_url']}")
            
            try:
                # 제품 상세 페이지로 이동
                self.driver.get(product['product_url'])
                # 필수 요소가 로드될 때까지만 대기 (최대 3초)
                try:
                    WebDriverWait(self.driver, 3).until(
                        EC.any_of(
                            EC.presence_of_element_located((By.CSS_SELECTOR, "img.thumbnail-item")),
                            EC.presence_of_element_located((By.CSS_SELECTOR, "li.css-1jelciw"))
                        )
                    )
                except:
                    # 대기 실패해도 계속 진행
                    time.sleep(1)  # 최소한의 대기
                
                print("image url extract start")
                # 추가 정보 수집
                product['image_urls'] = self.get_thumbnail_image()
                print("key features extract start")
                product['key_features'] = self.get_key_features()
                
                updated_products.append(product)
                print(product)
                
            except Exception as e:
                print(f" 상세 정보 수집 실패: {e}")
                # 실패해도 기본 정보는 유지
                product['image_urls'] = []
                product['key_features'] = []
                updated_products.append(product)
        
        return updated_products

    def get_thumbnail_image(self):
        """첫 번째 제품 이미지 URL 추출 (img src 사용)"""
        try:
            # thumbnail-item 클래스를 가진 img 태그의 src 속성에서 추출
            img = self.driver.find_element(By.CSS_SELECTOR, "img.thumbnail-item")
            src = img.get_attribute('src')
            
            if src and 'media.us.lg.com' in src:
                return src
            
            # 백업: 일반 img 태그에서
            img_backup = self.driver.find_element(By.CSS_SELECTOR, "img[src*='media.us.lg.com']")
            return img_backup.get_attribute('src')
            
        except Exception as e:
            print(f"    이미지 추출 실패: {e}")
            return "N/A"

    def get_key_features(self):
        """주요 특징들 추출"""
        features = []
        
        try:
            # li 요소들 찾기
            feature_items = self.driver.find_elements(By.CSS_SELECTOR, "li.css-1jelciw")
            
            for item in feature_items:
                try:
                    # li 안의 span 텍스트 추출
                    span = item.find_element(By.CSS_SELECTOR, "span")
                    feature_text = span.text.strip()
                    if feature_text:
                        features.append(feature_text)
                except:
                    # span이 없으면 li의 직접 텍스트 추출
                    feature_text = item.text.strip()
                    if feature_text:
                        features.append(feature_text)
            
            print(f"    주요 특징 {len(features)}개 추출")
            return features
            
        except Exception as e:
            print(f"    특징 추출 실패: {e}")
            return []
    
    def extract_size_number(self, size_text):
        """Extract numeric size from text like '83"' -> 83"""
        try:
            return int(size_text.replace('"', '').replace("'", ''))
        except ValueError:
            return 0  
    
    def parse_products(self):
        
        basic_products_info = []
    
        cards = self.driver.find_elements(By.CSS_SELECTOR, ".mh-product-box")
        
        print("\n1단계: 기본 제품 정보 수집")
        
        for i, card in enumerate(cards):
            print(f"카드 {i+1}/{len(cards)} 처리중...")
            
            # 사이즈 버튼들 찾기
            buttons = card.find_elements(By.CSS_SELECTOR, ".MuiChip-clickable")
            
            for button in buttons:
                try:
                    """Extract numeric size from text like '83"' -> 83"""
                    size_text = button.text.strip()
                    size = self.extract_size_number(size_text)
                    
                    # 버튼 클릭
                    button.click()
                    time.sleep(1)
                    
                    basic_product = self.extract_product_info(card, size)

                    if basic_product:
                        basic_products_info.append(basic_product)
                        # print(f"  {size}: {product_info['product_name']} - ${product_info['price']}")
                    
                except Exception as e:
                    print(f"버튼 처리 실패: {e}")
                    continue
                if len(basic_products_info) > 1:
                    print("\n2단계: 상세 정보 수집")
                    detailed_products_info = self.scrape_product_details(basic_products_info)
                                
                    # JSON 파일로 저장
                    self.save_products_to_json(detailed_products_info)
                    
                    print(f"\n총 {len(detailed_products_info)}개 제품 정보 수집 완료")
                    return detailed_products_info
        # 2. 각 제품의 상세 정보 수집
        print("\n2단계: 상세 정보 수집")
        detailed_products_info = self.scrape_product_details(basic_products_info)
                    
        # JSON 파일로 저장
        self.save_products_to_json(detailed_products_info)
        
        print(f"\n총 {len(detailed_products_info)}개 제품 정보 수집 완료")
        return detailed_products_info

    def scrape(self):
        self.open_page()
        self.click_view_all()
        self.click_load_more()
        self.parse_products()

    def save_products_to_json(self, products_info: list) -> None:
        """
        Add metadata structure to JSON output for better AI/search performance.
        
        Args:
            products_info (list): List of basic product dictionaries
        """
        final_data = {
        "metadata": {
            "units": {"price_currency": "USD", "size_unit": "inches"},
            "total_products": len(products_info)
        },
        "products": products_info
        }
        """제품 정보를 JSON 파일로 저장"""
        try:
            filename = f"{self.category}_products.json"
            
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(final_data, f, ensure_ascii=False, indent=2)
            
            print(f"제품 정보가 {filename}에 저장되었습니다. ({len(products_info)}개 제품)")
            
        except Exception as e:
            print(f"JSON 저장 실패: {e}")

    def close(self):
        self.driver.quit()

