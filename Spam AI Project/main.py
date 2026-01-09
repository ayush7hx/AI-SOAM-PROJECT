# main.py
"""
EMAIL SPAM DETECTOR AI - Main Application
Run this file to use the email spam detection system
"""

import os
import sys
from src.predict import SpamPredictor

def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def display_banner():
    """Display application banner"""
    print("="*50)
    print("     📧  EMAIL SPAM DETECTOR AI")
    print("    CBSE AI Project - Class X")
    print("         Made by: Ayush7hx")
    print("="*50)
    print()

def main_menu():
    """Display main menu"""
    print("\n" + "═"*50)
    print("� EMAIL SPAM DETECTOR MENU")
    print("═"*50)
    print("1. 📧 Check a single email")
    print("2. 📁 Check multiple emails from file")
    print("3. 📊 View model information")
    print("4. 🧪 Test with sample emails")
    print("5. ❌ Exit")
    print("═"*50)
    
    choice = input("\nEnter your choice (1-5): ")
    return choice

def check_single_message(predictor):
    """Check if a single email is spam"""
    clear_screen()
    print("\n" + "─"*50)
    print("📧 CHECK SINGLE EMAIL")
    print("─"*50)
    
    message = input("\nEnter the email content to check:\n> ")
    
    if not message.strip():
        print("❌ No email content entered!")
        return
    
    print("\n" + "─"*50)
    result = predictor.predict(message)
    
    if "error" in result:
        print(result["error"])
        return
    
    # Display result
    print(f"📩 Message: {result['message']}")
    print(f"✅ Prediction: {result['prediction']}")
    
    # Visual indicator
    if result['prediction'] == "SPAM":
        print("   ⚠️  WARNING: This appears to be SPAM!")
    else:
        print("   ✓ This appears to be legitimate (HAM)")
    
    print(f"📊 Confidence: {result['confidence']:.2%}")
    print(f"   - HAM probability: {result['probabilities']['HAM']:.2%}")
    print(f"   - SPAM probability: {result['probabilities']['SPAM']:.2%}")
    
    input("\nPress Enter to continue...")

def check_multiple_messages(predictor):
    """Check multiple emails from a file"""
    clear_screen()
    print("\n" + "─"*50)
    print("📁 CHECK MULTIPLE EMAILS")
    print("─"*50)
    
    filename = input("\nEnter filename (txt file, one email per line): ")
    
    if not os.path.exists(filename):
        print(f"❌ File '{filename}' not found!")
        input("\nPress Enter to continue...")
        return
    
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            messages = [line.strip() for line in file if line.strip()]
        
        print(f"\n📬 Found {len(messages)} emails")
        print("\nResults:")
        print("─"*50)
        
        spam_count = 0
        for i, msg in enumerate(messages[:20], 1):  # Limit to 20 for display
            result = predictor.predict(msg)
            if "error" not in result:
                indicator = "🔴" if result['prediction'] == "SPAM" else "🟢"
                print(f"{i:2}. {indicator} {result['prediction']:4} - {msg[:50]}...")
                if result['prediction'] == "SPAM":
                    spam_count += 1
        
        print("─"*50)
        print(f"\n📊 Summary: {spam_count} spam emails, {len(messages)-spam_count} ham emails")
        
        if len(messages) > 20:
            print(f"   (Showing first 20 of {len(messages)} emails)")
    
    except Exception as e:
        print(f"❌ Error reading file: {e}")
    
    input("\nPress Enter to continue...")

def view_model_info():
    """Display model information"""
    clear_screen()
    print("\n" + "─"*50)
    print("📊 EMAIL SPAM DETECTOR MODEL INFO")
    print("─"*50)
    
    print("\n🔧 Technical Details:")
    print("   • Algorithm: Naive Bayes (Multinomial)")
    print("   • Features: 5,000 most common words")
    print("   • Dataset: Email messages")
    print("   • Accuracy: ~98% (on test data)")
    
    print("\n📈 Training Data:")
    print("   • Total emails: 28 messages")
    print("   • Ham (legitimate): 10 emails (36%)")
    print("   • Spam: 18 emails (64%)")
    
    print("\n🎯 How it works:")
    print("   1. Cleans email text (removes URLs, emails, phones)")
    print("   2. Converts words to numerical features")
    print("   3. Uses probability to classify as spam/ham")
    
    input("\nPress Enter to continue...")

def test_samples(predictor):
    """Test with sample emails"""
    clear_screen()
    print("\n" + "─"*50)
    print("🧪 TEST WITH SAMPLE EMAILS")
    print("─"*50)
    
    samples = [
        ("Subject: Meeting Tomorrow Hi team, Just a reminder about our meeting scheduled for tomorrow at 2 PM in Conference Room A.", "HAM"),
        ("WINNER! You've won $1,000,000! Click here to claim your prize: http://fake-lottery.com/claim", "SPAM"),
        ("Dear customer, Your order #12345 has been shipped. Track it here: http://tracking.company.com", "HAM"),
        ("URGENT: Your PayPal account is suspended! Verify now: http://paypal-secure-login.com", "SPAM"),
        ("Hi Mom, I'll be home late tonight. Traffic is bad. Love you!", "HAM"),
        ("Congratulations! You've won a free iPhone.", "SPAM"),
        ("Mom, can you pick me up at 5 pm?", "HAM"),
        ("URGENT: Your account has been compromised", "SPAM"),
        ("Reminder: Meeting at 3 PM in conference room", "HAM"),
        ("You have won a lottery! Claim your prize", "SPAM"),
        ("Thanks for your help yesterday", "HAM")
    ]
    
    print("\nTesting predictions:")
    print("─"*50)
    
    correct = 0
    for msg, expected in samples:
        result = predictor.predict(msg)
        if "error" not in result:
            is_correct = result['prediction'] == expected
            if is_correct:
                correct += 1
            
            symbol = "✅" if is_correct else "❌"
            print(f"{symbol} '{msg[:40]}...'")
            print(f"   Predicted: {result['prediction']} | Expected: {expected}")
            print(f"   Confidence: {result['confidence']:.2%}")
            print()
    
    accuracy = correct / len(samples) * 100
    print("─"*50)
    print(f"📊 Test Accuracy: {accuracy:.1f}% ({correct}/{len(samples)} correct)")
    
    input("\nPress Enter to continue...")

def main():
    """Main application function"""
    clear_screen()
    display_banner()
    
    print("Loading AI model...")
    predictor = SpamPredictor()
    
    if predictor.model is None:
        print("\n❌ Failed to load model. Please train the model first.")
        print("   Run: python src/train_model.py")
        input("\nPress Enter to exit...")
        return
    
    print("✅ Model loaded successfully!\n")
    
    while True:
        clear_screen()
        display_banner()
        
        choice = main_menu()
        
        if choice == '1':
            check_single_message(predictor)
        elif choice == '2':
            check_multiple_messages(predictor)
        elif choice == '3':
            view_model_info()
        elif choice == '4':
            test_samples(predictor)
        elif choice == '5':
            print("\n👋 Thank you for using Spam Detector AI!")
            print("   Goodbye! 👋")
            break
        else:
            print("\n❌ Invalid choice! Please enter 1-5.")
            input("Press Enter to continue...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Program interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        input("Press Enter to exit...")