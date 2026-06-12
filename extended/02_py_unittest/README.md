pip install pytest pytest-html coverage

```
coverage run -m pytest -v tests/test_wf.py --html=pytest_report.html --self-contained-html
coverage report -m
coverage html
```