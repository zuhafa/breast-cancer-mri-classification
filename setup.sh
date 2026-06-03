#!/bin/bash

# Breast Cancer MRI Classification Platform - Setup Script
# =========================================================

set -e

echo "=============================================="
echo "Breast Cancer MRI Classifier - Setup"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Python version
check_python() {
    print_status "Checking Python version..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
        print_success "Python version: $PYTHON_VERSION"
    elif command -v python &> /dev/null; then
        PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
        print_success "Python version: $PYTHON_VERSION"
    else
        print_error "Python is not installed. Please install Python 3.9 or higher."
        exit 1
    fi
}

# Check Node.js version
check_node() {
    print_status "Checking Node.js version..."
    
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version)
        print_success "Node.js version: $NODE_VERSION"
    else
        print_error "Node.js is not installed. Please install Node.js 18 or higher."
        exit 1
    fi
}

# Setup backend
setup_backend() {
    print_status "Setting up backend..."
    
    cd backend
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv venv || python -m venv venv
    fi
    
    # Activate virtual environment
    print_status "Activating virtual environment..."
    source venv/bin/activate || source venv/Scripts/activate
    
    # Upgrade pip
    print_status "Upgrading pip..."
    pip install --upgrade pip
    
    # Install requirements
    print_status "Installing Python dependencies..."
    pip install -r requirements.txt
    
    # Create models directory
    if [ ! -d "models" ]; then
        print_status "Creating models directory..."
        mkdir -p models
    fi
    
    # Create uploads directory
    if [ ! -d "uploads" ]; then
        print_status "Creating uploads directory..."
        mkdir -p uploads
    fi
    
    print_success "Backend setup complete!"
    cd ..
}

# Setup frontend
setup_frontend() {
    print_status "Setting up frontend..."
    
    cd frontend
    
    # Check if node_modules exists
    if [ ! -d "node_modules" ]; then
        print_status "Installing Node.js dependencies..."
        npm install
    else
        print_warning "node_modules already exists. Skipping npm install."
    fi
    
    print_success "Frontend setup complete!"
    cd ..
}

# Create environment files
create_env_files() {
    print_status "Creating environment files..."
    
    # Backend .env
    if [ ! -f "backend/.env" ]; then
        cp backend/.env.example backend/.env
        print_success "Created backend/.env"
    fi
    
    # Frontend .env
    if [ ! -f "frontend/.env" ]; then
        echo "VITE_API_URL=http://localhost:8000/api/v1" > frontend/.env
        print_success "Created frontend/.env"
    fi
}

# Print next steps
print_next_steps() {
    echo ""
    echo "=============================================="
    echo "Setup Complete!"
    echo "=============================================="
    echo ""
    echo "Next Steps:"
    echo ""
    echo "1. Place your trained model files in backend/models/"
    echo "   - resnet_model.keras"
    echo "   - densenet_model.keras"
    echo "   - efficientnet_model.keras"
    echo "   - convnext_model.keras"
    echo ""
    echo "2. Start the backend server:"
    echo "   cd backend"
    echo "   source venv/bin/activate  # Linux/Mac"
    echo "   venv\\Scripts\\activate    # Windows"
    echo "   uvicorn app.main:app --reload"
    echo ""
    echo "3. Start the frontend (in a new terminal):"
    echo "   cd frontend"
    echo "   npm run dev"
    echo ""
    echo "4. Open your browser and navigate to:"
    echo "   http://localhost:5173"
    echo ""
    echo "=============================================="
}

# Main execution
main() {
    print_status "Starting setup..."
    
    check_python
    check_node
    setup_backend
    setup_frontend
    create_env_files
    
    print_next_steps
}

# Run main function
main
