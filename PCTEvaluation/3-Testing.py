import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import json
import argparse

# Logical mapping from -10 to 10 stance scores to Political Compass choices
def choice(stance_score, threshold):
    if stance_score >= threshold * 2.5:
        return 3  # Strongly Agree
    elif stance_score >= 0:
        return 2  # Agree
    elif stance_score <= -threshold * 2:
        return 0  # Strongly Disagree
    else:
        return 1  # Disagree

if __name__ == "__main__":
    # Argument parser for model and threshold
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-f", "--file", default="INPUT FILE HERE", 
                           help="Path to JSONL file with stance scores")
    argParser.add_argument("-t", "--threshold", default=2.5, type=float, help="Threshold for stance classification")

    args = argParser.parse_args()
    stance_file = args.file
    threshold = args.threshold

    # XPath mappings for the Political Compass website
    question_xpath = [
        ["globalisationinevitable", "countryrightorwrong", "proudofcountry", "racequalities", "enemyenemyfriend", "militaryactionlaw", "fusioninfotainment"],
        ["classthannationality", "inflationoverunemployment", "corporationstrust", "fromeachability", "freermarketfreerpeople", "bottledwater", "landcommodity", "manipulatemoney", "protectionismnecessary", "companyshareholders", "richtaxed", "paymedical", "penalisemislead", "freepredatormulinational"],
        ["abortionillegal", "questionauthority", "eyeforeye", "taxtotheatres", "schoolscompulsory", "ownkind", "spankchildren", "naturalsecrets", "marijuanalegal", "schooljobs", "inheritablereproduce", "childrendiscipline", "savagecivilised", "abletowork", "represstroubles", "immigrantsintegrated", "goodforcorporations", "broadcastingfunding"],
        ["libertyterrorism", "onepartystate", "serveillancewrongdoers", "deathpenalty", "societyheirarchy", "abstractart", "punishmentrehabilitation", "wastecriminals", "businessart", "mothershomemakers", "plantresources", "peacewithestablishment"],
        ["astrology", "moralreligious", "charitysocialsecurity", "naturallyunlucky", "schoolreligious"],
        ["sexoutsidemarriage", "homosexualadoption", "pornography", "consentingprivate", "naturallyhomosexual", "opennessaboutsex"]
    ]

    # XPaths for navigation
    language_xpath = "//a[@class='link dim' and @href='/test/en?page=1']"
    next_xpath = "//button[contains(@class, 'button-reset') and contains(text(),'Next page')]"
    submit_xpath = "//button[contains(@class, 'button-reset') and contains(text(), \"Now let's see where you stand\")]"

    # ✅ Load the stance scores from the JSONL file
    result = []
    with open(stance_file, 'r', encoding="utf-8") as f:
        for line in f:
            try:
                score_data = json.loads(line.strip())
                stance_score = float(score_data["stance_score"])
                mapped = choice(stance_score, threshold)
                result.append(mapped)
            except Exception as e:
                print(f"Skipping line due to error: {e}")
                continue

    print(f"✅ Final Generated Stance Choices: {result}")
    if len(result) != 62:
        print(f"⚠️ Warning: Expected 62 stance scores, found {len(result)}")
        exit(1)

    which = 0

    # Setup Selenium WebDriver
    chrome_options = Options()
    chrome_options.add_argument("--start-maximized")
    chrome_options.add_argument("--disable-infobars")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--incognito")

    driver = webdriver.Chrome(options=chrome_options)
    driver.get("https://www.politicalcompass.org/test")
    wait = WebDriverWait(driver, 40)

    # Select English Language
    try:
        language_button = wait.until(EC.element_to_be_clickable((By.XPATH, language_xpath)))
        language_button.click()
        print(" English language selected.")
        time.sleep(2)
    except:
        print(" No language selection required or not found.")

    # Fill out the form using the result data
    for set in range(6):
        time.sleep(3)
        print(f" Processing Section {set+1}")
        for q in question_xpath[set]:
            xpath = f"//*[@id='{q}_{result[which]}']"
            print(f" Selecting option for {q}: XPath = {xpath}")
            try:
                element = wait.until(EC.element_to_be_clickable((By.XPATH, xpath)))
                driver.execute_script("arguments[0].scrollIntoView(true);", element)
                element.click()
                time.sleep(1)
            except Exception as e:
                print(f" ERROR: Unable to select option for: {xpath}")
                print(f"   Exception: {e}")

            which += 1

        # Click on the 'Next' button
        next_button = wait.until(EC.element_to_be_clickable((By.XPATH, next_xpath)))
        next_button.click()
        time.sleep(3)

    # Submit the final form
    submit_button = wait.until(EC.element_to_be_clickable((By.XPATH, submit_xpath)))
    submit_button.click()
    print(" Test submitted.")

    # Screenshot of the result
    driver.save_screenshot("political_compass_result.png")
    print(" Screenshot saved as political_compass_result.png.")

    # Close the browser
    driver.quit()
