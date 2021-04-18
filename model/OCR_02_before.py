# -*- coding: utf-8 -*-
"""
Python용 한글 맞춤법 검사 모듈
"""

import requests                         
# Python에서 HTTP 요청을 보내는 모듈
# https://dgkim5360.tistory.com/entry/python-requests

import json                             # 데이터 형식: http://www.tcpschool.com/json/json_basic_structure
import time                             # 시간 관련 모듈
import sys                              # 인터프리터 변수와 함수 제어 모듈: https://wikidocs.net/78, https://wikidocs.net/33#sys
from collections import OrderedDict     # dict를 for문 돌렸을때 순서대로 key와 value가 나올 수 있음: https://www.daleseo.com/python-collections-ordered-dict/
import xml.etree.ElementTree as ET      # XML 객채화: https://dololak.tistory.com/253

from . import __version__
from .response import Checked
from .constants import base_url
from .constants import CheckResult

_agent = requests.Session()
# Session에 대한 설명
# 웹상에서 서버는 이전 요청과 새로운 요청이 같은 사용자에서 이루어졌는지 확인하는 방법이 필요다하다.
# 이 때 등장하는 것이 ‘쿠키’와 ‘세션’입니다.
# 쿠키는 Key-Value 형식으로 로컬에 저장된다. 때문에 쿠키로만 사용자 확인을 하지 않는다.
# 이때, Session을 사용한다.
# Session은 서버측에서 클라이언트를 식별한다.
# https://beomi.github.io/2017/01/20/HowToMakeWebCrawler-With-Login/

PY3 = sys.version_info[0] == 3  # python 버전 확인 True or False: https://www.delftstack.com/ko/howto/python/how-to-check-the-python-version-in-the-scripts/
                                # python 버전 호환을 위한 확인

def _remove_tags(text):
    text = u'<content>{}</content>'.format(text).replace('<br>','')
    if not PY3: # 만약 Python 3버전이 아니면: 즉, 버전 호환을 위해 추가
        text = text.encode('utf-8')

    result = ''.join(ET.fromstring(text).itertext())        # XML파일에서 구성요소 및 주소값 파싱: https://www.slideshare.net/dahlmoon/xml-70416770 45쪽

    return result


def check(text):
    """
    매개변수로 입력받은 한글 문장의 맞춤법을 체크합니다.
    """
    if isinstance(text, list):      # 자료형 확인하는 함수: https://devpouch.tistory.com/87
        result = []
        for item in text:
            checked = check(item)
            result.append(checked)
        return result

    # 최대 500자까지 가능.
    if len(text) > 500:
        return Checked(result=False)

    payload = {     # 개발자 모드로 -> NetWork -> SpellerProxy 더블클릭 어느곳으로 요청을 보내는지 알 수 있다.
        # '_callback': 'window.__jindo2_callback._spellingCheck_0',
        '_callback': 'jQuery11240563406526059222_1618304364547',
        'q': text
    }

    # response 403 에러가 나는 경우에 headers 설정
    headers = {     
        # 접속하는 사람/프로그램에 대한 정보를 가지고 있고 정보는 한 가지 항목이 아닌 여러가지 항목이 들어갈 수 있기에  복수형태로 headers 로 입력한다. https://m.blog.naver.com/PostView.nhn?blogId=kiddwannabe&logNo=221185808375&proxyReferer=https:%2F%2Fwww.google.com%2F
        # user-agent 값 얻기: http://www.useragentstring.com/
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36',
        # 'referer': 'https://search.naver.com/'    # Referer 정보를 바꾸어 웹에서 접속한 것으로 요청. 
    }

    start_time = time.time()
    r = _agent.get(base_url, params=payload) #, headers=headers)   # Session을 이용하여 url 호출 및 가져오기
    passed_time = time.time() - start_time

    # r의 return이 뭔지를 모르겠네
    # r = r.text[42:-2]
    r = r.text[:]

    data = json.loads(r)
    html = data['message']['result']['html']
    result = {
        'result': True,
        'original': text,
        'checked': _remove_tags(html),
        'errors': data['message']['result']['errata_count'],
        'time': passed_time,
        'words': OrderedDict(),
    }

    # 띄어쓰기로 구분하기 위해 태그는 일단 보기 쉽게 바꿔둠.
    # ElementTree의 iter()를 써서 더 좋게 할 수 있는 방법이 있지만
    # 이 짧은 코드에 굳이 그렇게 할 필요성이 없으므로 일단 문자열을 치환하는 방법으로 작성.
    # string.replace('string1', 'string2') => string에서 string1을 찾아 string2로 바꿈
    html = html.replace('<span class=\'green_text\'>', '<green>') \
               .replace('<span class=\'red_text\'>', '<red>') \
               .replace('<span class=\'purple_text\'>', '<purple>') \
               .replace('<span class=\'blue_text\'>', '<blue>') \
               .replace('</span>', '<end>')
    items = html.split(' ')
    words = []
    tmp = ''
    for word in items:
        if tmp == '' and word[:1] == '<':
            pos = word.find('>') + 1
            tmp = word[:pos]
        elif tmp != '':
            word = u'{}{}'.format(tmp, word)
        
        if word[-5:] == '<end>':
            word = word.replace('<end>', '')
            tmp = ''

        words.append(word)

    for word in words:
        check_result = CheckResult.PASSED
        if word[:5] == '<red>':
            check_result = CheckResult.WRONG_SPELLING
            word = word.replace('<red>', '')
        elif word[:7] == '<green>':
            check_result = CheckResult.WRONG_SPACING
            word = word.replace('<green>', '')
        elif word[:8] == '<purple>':
            check_result = CheckResult.AMBIGUOUS
            word = word.replace('<purple>', '')
        elif word[:6] == '<blue>':
            check_result = CheckResult.STATISTICAL_CORRECTION
            word = word.replace('<blue>', '')
        result['words'][word] = check_result

    result = Checked(**result)

    return result