//
//  TeamMembersViewController.swift
//  dermCheckAI
//
//  Created by Ruby Han on 18/11/2022.
//  Copyright © 2022 AppCoda. All rights reserved.
//

import UIKit

class TeamMembersViewController: UIViewController {

    @IBOutlet weak var collectionView: UICollectionView!
    @IBOutlet weak var pageControl: UIPageControl!
    
    var slides: [TeamMemberSlide] = []
    
    var currentPage = 0
    
    override func viewDidLoad() {
        super.viewDidLoad()

        // Do any additional setup after loading the view.
        // collectionView.delegate = self
        // collectionView.dataSource = self
        
        slides = [
            TeamMemberSlide(title: "Ruby Han", description: "Modeling & Analytics Engineer / Data Scientist", image: #imageLiteral(resourceName: "ruby_han")),
            TeamMemberSlide(title: "George Jiang", description: "Data Scientist", image: #imageLiteral(resourceName: "george_jiang")),
            TeamMemberSlide(title: "Gerrit Lensink", description: "Data Scientist", image: #imageLiteral(resourceName: "gerrit_lensink")),
            TeamMemberSlide(title: "Shivani Sharma", description: "Senior Consultant", image: #imageLiteral(resourceName: "shivani_sharma"))
            
        ]
    }
    

}

extension TeamMembersViewController:
    UICollectionViewDelegate,
    UICollectionViewDataSource, UICollectionViewDelegateFlowLayout {
    
    func collectionView(_ collectionView: UICollectionView, numberOfItemsInSection section: Int) -> Int {
        return slides.count
    }
    
    func collectionView(_ collectionView: UICollectionView, cellForItemAt indexPath: IndexPath) -> UICollectionViewCell {
        let cell =
            collectionView.dequeueReusableCell(withReuseIdentifier: TeamMembersCollectionViewCell.identifier, for: indexPath)
            as! TeamMembersCollectionViewCell
        cell.setup(slides[indexPath.row])
        return cell
    }
    
    func collectionView(_ collectionView: UICollectionView, layout collectionViewLayout: UICollectionViewLayout, sizeForItemAt indexPath: IndexPath) -> CGSize {
    return CGSize(width: collectionView.frame.width, height: collectionView.frame.height)
    }
    func scrollViewDidEndDecelerating(_ scrollView: UIScrollView) {
        let width = scrollView.frame.width
        currentPage = Int(scrollView.contentOffset.x / width)
        pageControl.currentPage = currentPage
    }
}

