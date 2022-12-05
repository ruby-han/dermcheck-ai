//
//  TeamMembersCollectionViewCell.swift
//  dermCheckAI
//
//  Created by Ruby Han on 18/11/2022.
//  Copyright © 2022 AppCoda. All rights reserved.
//

import UIKit

class TeamMembersCollectionViewCell: UICollectionViewCell {
    
    static let identifier = String(describing: TeamMembersCollectionViewCell.self)
    
    @IBOutlet weak var slideImageView: UIImageView!
    @IBOutlet weak var slideTitleLbl: UILabel!
    @IBOutlet weak var slideDescriptionLbl: UILabel!
    
    func setup(_ slide: TeamMemberSlide){
        slideImageView.image = slide.image
        slideTitleLbl.text = slide.title
        slideDescriptionLbl.text = slide.description
    }
}
