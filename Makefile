# you may have to run `aws --profile=bsq-test ecr get-login --no-include-email` before doing a push

.PHONY: all

all: test, build

install:
	pip install -r requirements.txt

install-ci:
	sudo pip install -r requirements-ci.txt

test:
	pytest -vv

test-watch:
	pytest-watch -v -- -vv

test-watch-notify:
	pytest-watch -v \
		--onpass "notify-send -u normal \"All tests passed!\"" \
		--onfail "notify-send -u critical \"Tests failed\"" \
		-- -vv
