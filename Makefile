test:
	@clear
	python -c  "import crab;crab.test()"

testcover:
	@clear
	python -c "import crab;crab.test(coverage=True)"
