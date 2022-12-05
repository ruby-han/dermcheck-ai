//
//  AppointmentConfirmVC.swift
//  dermCheckAI
//
//  Created by Ruby Han on 17/11/22.
//

import UIKit
import Lottie

class AppointmentConfirmVC: UIViewController {

    @IBOutlet weak var animationViewLottie: LOTAnimatedControl!
    override func viewDidLoad() {
        super.viewDidLoad()

        showOrderCompleteAnimation()
    }
    
    private func showOrderCompleteAnimation() {
        // Checkmark animation setup
        animationViewLottie.animationView.setAnimation(named: "Check_Mark")
        animationViewLottie.contentMode = .scaleAspectFit
        animationViewLottie.animationView.loopAnimation = false
        
        animationViewLottie.animationView.play()
    }

}
