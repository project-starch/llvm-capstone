#!/usr/bin/env python3
"""Run one Speedometer in Firefox under Selenium and print its score.

usage: ff-speedometer.py FIREFOX GECKODRIVER URL VERSION LOG_PREFIX
VERSION 2.1 starts the run through the page's own startTest() (its index has no automatic start); 3.1 starts by
itself (?startAutomatically in URL). Done is the score in #result-number, which both versions fill only when the
run completes. Exit 0 with "score <value>" on stdout, 1 if no score appeared within an hour."""
import sys
import time

from selenium import webdriver
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.firefox.service import Service


def main():
    firefox, geckodriver, url, version, log = sys.argv[1:6]
    opts = Options()
    opts.binary_location = firefox
    opts.add_argument("-headless")
    driver = webdriver.Firefox(options=opts, service=Service(executable_path=geckodriver,
                                                               log_output=log + ".geckodriver.log"))
    try:
        driver.set_page_load_timeout(300)
        driver.get(url)
        if version == "2.1":
            time.sleep(3)
            driver.execute_script("startTest()")
        score, deadline = "", time.time() + 3600
        while time.time() < deadline:
            score = driver.execute_script(
                "var e = document.getElementById('result-number'); return e ? e.textContent : '';").strip()
            if score:
                break
            time.sleep(5)
        print(f"score {score}" if score else "no score")
        return 0 if score else 1
    finally:
        driver.quit()


if __name__ == "__main__":
    sys.exit(main())
