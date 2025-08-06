# Makefile for RAG Document Q&A System Testing

.PHONY: help test test-unit test-integration test-all coverage lint format clean install-test-deps check-env

# Default target
help:
	@echo "Available targets:"
	@echo "  help              - Show this help message"
	@echo "  install-test-deps - Install testing dependencies"
	@echo "  check-env         - Check test environment setup"
	@echo "  test              - Run unit tests (default)"
	@echo "  test-unit         - Run unit tests only"
	@echo "  test-integration  - Run integration tests only"
	@echo "  test-all          - Run all tests"
	@echo "  test-fast         - Run unit tests without coverage"
	@echo "  test-specific     - Run specific test (usage: make test-specific TEST=path/to/test)"
	@echo "  coverage          - Generate coverage report"
	@echo "  lint              - Run code linting"
	@echo "  format            - Format code with black and isort"
	@echo "  clean             - Clean test artifacts"
	@echo "  test-watch        - Run tests in watch mode"
	@echo ""
	@echo "Examples:"
	@echo "  make test                                    # Run unit tests"
	@echo "  make test-specific TEST=tests/unit/test_app.py  # Run specific test"
	@echo "  make coverage                                # Generate coverage report"

# Install testing dependencies
install-test-deps:
	@echo "Installing testing dependencies..."
	pip install -r requirements-test.txt

# Check test environment
check-env:
	@echo "Checking test environment..."
	python run_tests.py --check-env

# Run unit tests (default)
test: test-unit

# Run unit tests
test-unit:
	@echo "Running unit tests..."
	python run_tests.py --unit --verbose

# Run integration tests
test-integration:
	@echo "Running integration tests..."
	python run_tests.py --integration --verbose

# Run all tests
test-all:
	@echo "Running all tests..."
	python run_tests.py --all --verbose

# Run unit tests without coverage (faster)
test-fast:
	@echo "Running unit tests (fast mode)..."
	python run_tests.py --unit --no-coverage --verbose

# Run specific test
test-specific:
	@echo "Running specific test: $(TEST)"
	python run_tests.py --test $(TEST) --verbose

# Generate coverage report
coverage:
	@echo "Generating coverage report..."
	python run_tests.py --coverage

# Run linting
lint:
	@echo "Running code linting..."
	python run_tests.py --lint

# Format code
format:
	@echo "Formatting code..."
	black src/ tests/ --line-length 100
	isort src/ tests/ --profile black

# Clean test artifacts
clean:
	@echo "Cleaning test artifacts..."
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf test_report.json
	rm -rf __pycache__/
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

# Run tests in watch mode (requires pytest-watch)
test-watch:
	@echo "Running tests in watch mode..."
	@if command -v ptw >/dev/null 2>&1; then \
		ptw --runner "python -m pytest tests/unit/ -v"; \
	else \
		echo "pytest-watch not installed. Install with: pip install pytest-watch"; \
		echo "Falling back to single run..."; \
		make test-unit; \
	fi

# Run tests with different Python versions (requires pyenv)
test-multi-python:
	@echo "Testing with multiple Python versions..."
	@for version in 3.8.18 3.9.18 3.10.13 3.11.8; do \
		echo "Testing with Python $$version..."; \
		if pyenv versions | grep -q $$version; then \
			pyenv shell $$version && python run_tests.py --unit --no-coverage; \
		else \
			echo "Python $$version not available, skipping..."; \
		fi; \
	done

# Run security checks
security:
	@echo "Running security checks..."
	@if command -v safety >/dev/null 2>&1; then \
		safety check; \
	else \
		echo "safety not installed. Install with: pip install safety"; \
	fi
	@if command -v bandit >/dev/null 2>&1; then \
		bandit -r src/; \
	else \
		echo "bandit not installed. Install with: pip install bandit"; \
	fi

# Run performance tests
test-performance:
	@echo "Running performance tests..."
	python -m pytest tests/ -m "slow" --benchmark-only --benchmark-sort=mean

# Generate test documentation
test-docs:
	@echo "Generating test documentation..."
	python -m pytest --collect-only --quiet | grep "test session starts" -A 1000 > test_inventory.txt
	@echo "Test inventory saved to test_inventory.txt"

# Continuous integration simulation
ci:
	@echo "Running CI pipeline simulation..."
	make clean
	make check-env
	make lint
	make test-all
	make coverage
	@echo "CI pipeline completed successfully!"

# Setup development environment
setup-dev:
	@echo "Setting up development environment..."
	pip install -r requirements.txt
	pip install -r requirements-test.txt
	@echo "Development environment setup complete!"

# Quick test (for development)
quick:
	@echo "Running quick tests..."
	python -m pytest tests/unit/ -x -v --tb=short

# Verbose test with debug info
debug:
	@echo "Running tests with debug information..."
	python -m pytest tests/unit/ -v --tb=long --capture=no -s

# Test specific module
test-module:
	@echo "Testing module: $(MODULE)"
	python -m pytest tests/unit/test_$(MODULE).py -v

# Test with coverage and open HTML report
test-coverage-open:
	make coverage
	@if command -v open >/dev/null 2>&1; then \
		open htmlcov/index.html; \
	elif command -v xdg-open >/dev/null 2>&1; then \
		xdg-open htmlcov/index.html; \
	else \
		echo "Coverage report generated at htmlcov/index.html"; \
	fi